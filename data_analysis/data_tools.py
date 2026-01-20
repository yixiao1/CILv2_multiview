import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from os.path import commonpath
from collections import defaultdict

# os.environ['FORCE_TF_AVAILABLE'] = '1'

from glob import glob
from typing import Union, List, Tuple, Type, Dict
from tqdm import tqdm
import click
from dataloaders import transforms
from PIL import Image
import math
import cv2
from multiprocessing import Pool
from _utils import utils, eval_utils
import json
import shutil
import re
from einops import rearrange

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import torch.nn.functional as F
from collections import deque
from pathlib import Path

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from transformers.models.mask2former import modeling_mask2former, image_processing_mask2former
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from network.models.architectures.CIL_multiview.CIL_multiview import CIL_multiview
from dataloaders.transforms import canbus_normalization, train_transform
from configs import g_conf
from dataloaders.transforms import decode_onehot_directions_to_str

# ====================== Helper functions ======================


def is_json_corrupted(filepath):
    """Check if a JSON file is corrupted by attempting to parse it."""
    try:
        with open(filepath, 'r') as file:
            json.load(file)
        return False
    except json.JSONDecodeError:
        return True  # Returns True if the JSON is corrupted


def is_image_corrupted(filepath):
    """Check if an image file is corrupted by attempting to read it."""
    img = cv2.imread(filepath)
    return img is None  # Returns True if the image is corrupted (i.e., cannot be read)


def get_paths(data_root: str, sensors: list = None) -> list:
    # Let's get all the paths for ALL the files in the dataset
    paths = glob(os.path.join(data_root, '**', '*'), recursive=True)
    # Filter out with the sensors + only files
    if sensors is not None:
        paths = [path for path in paths if any(os.path.basename(path).startswith(sensor) for sensor in sensors) and os.path.isfile(path)]
    # We might have to filter out the noise images if we are not using them
    if 'virtual_attention' in sensors:
        paths = [path for path in paths if 'noise' not in path]
    # Sort the paths
    return sorted(paths)


def find_deepest_directories(start_path: str) -> List[str]:
    """Recursively find the deepest directories within the given start path."""
    deepest = []
    for root, dirs, files in os.walk(start_path):
        if not dirs:  # If there are no subdirectories, this is a deepest directory
            deepest.append(root)
    return deepest


def prepare_semantic_segmentation(args) -> type(None):
    """ Check if the semantic segmentation images only have the data in the red channel, so change it to RGB. """
    path, dataset, subdata, route = args
    # Open the image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f'Failed to load image: {path}')
        import sys; sys.exit(1)

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    # Check if all info is in one channel (only classes)
    if max(img[:, :, :3].flatten()) <= max(transforms.ss_classes):
        for k, v in transforms.ss_classes.items():
            # img is RGBA, so we need to check the first channel
            # Replace the R, G, and B values with those found in the dictionary
            mask = img[:, :, 0] == k
            img[mask, :3] = v

        # Save it (overwrite)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))


def check_single_ss_image(filepath: str) -> tuple:
    """Check if a single semantic segmentation image has class indices only in the red channel."""
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f'Failed to load image: {filepath}')
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    
    # Check if green and blue channels contain any non-zero values
    if np.any(img[:, :, 1]) or np.any(img[:, :, 2]):  # If any non-zero values in G or B
        max_values = [
            np.max(img[:, :, 0]),  # max value in R channel
            np.max(img[:, :, 1]),  # max value in G channel
            np.max(img[:, :, 2])   # max value in B channel
        ]
        return (filepath, max_values)
    return None


def check_semantic_segmentation(directory: str) -> list:
    """Check semantic segmentation images using multiprocessing."""
    # Get all files starting with 'ss'
    filepaths = [str(p) for p in Path(directory).rglob('ss*.png')]
    # Use all available CPU cores except one
    num_processes = max(1, cpu_count() - 1)
    # Create a pool of workers
    with Pool(processes=num_processes) as pool:
        # Map the check_single_image function to all filepaths
        results = pool.map(check_single_image, filepaths)
    # Filter out None results and return suspicious images
    return [r for r in results if r is not None]


def analyze_suspicious_paths(suspicious_images: list) -> None:
    """
    Analyze and group suspicious semantic segmentation image paths to find common patterns.
    Hence, use check_semantic_segmentation(PATH) above, then pass that list to this function.
    """
    if not suspicious_images:
        print("No suspicious images found.")
        return
        
    # Get all paths
    paths = [path for path, _ in suspicious_images]
    
    # Find the common root path for all suspicious images
    common_root = commonpath(paths)
    print(f"\nCommon root path: {common_root}")
    
    # Group paths by their directory structure
    path_groups = defaultdict(list)
    for path in paths:
        # Get relative path from common root
        rel_path = os.path.relpath(os.path.dirname(path), common_root)
        path_groups[rel_path].append(path)
    
    # Print grouped results
    print(f"\nFound {len(suspicious_images)} suspicious images in {len(path_groups)} directories:")
    for dir_path, files in path_groups.items():
        print(f"\nDirectory: {dir_path}")
        print(f"Count: {len(files)}")
        # Print a few example files
        if len(files) > 3:
            print("Example files:")
            for f in files[:3]:
                print(f"  - {os.path.basename(f)}")
            print(f"  ... and {len(files)-3} more")
        else:
            print("Files:")
            for f in files:
                print(f"  - {os.path.basename(f)}")


def get_frame_number(filepath):
    """Extract frame number from filepath."""
    match = re.search(r'(\d{6})', filepath)
    return int(match.group(1)) if match else None

def process_map(args) -> None:
    idx, noise_cat, depth_paths, semantic_segmentation_paths, depth_threshold, min_depth, num_data_route, base_path, route, converter_label = args
    
    # Get the frame number from the input files
    frame_number = get_frame_number(semantic_segmentation_paths[idx])
    if frame_number is None:
        print(f"Warning: Could not extract frame number from {semantic_segmentation_paths[idx]}")
        return
        
    *_, mask_merge_central = transforms.get_virtual_attention_map(
        depth_path=depth_paths[idx],
        segmented_path=semantic_segmentation_paths[idx],
        noise_cat=noise_cat,
        depth_threshold=depth_threshold,
        min_depth=min_depth,
        central_camera=True,
        converter_label=converter_label
    )
    *_, mask_merge_left = transforms.get_virtual_attention_map(
        depth_path=depth_paths[idx + num_data_route],
        segmented_path=semantic_segmentation_paths[idx + num_data_route],
        noise_cat=noise_cat,
        depth_threshold=depth_threshold,
        min_depth=min_depth,
        converter_label=converter_label
    )
    *_, mask_merge_right = transforms.get_virtual_attention_map(
        depth_path=depth_paths[idx + num_data_route * 2],
        segmented_path=semantic_segmentation_paths[idx + num_data_route * 2],
        noise_cat=noise_cat,
        depth_threshold=depth_threshold,
        min_depth=min_depth,
        converter_label=converter_label
    )

    # Set the name of the virtual attention files using the extracted frame number
    fname_central = f'virtual_attention_central_'
    fname_left    = f'virtual_attention_left_'
    fname_right   = f'virtual_attention_right_'
    
    # Add the noise, if the noise category is different from 0 (no noise)
    fname_central = f'{fname_central}noise_{noise_cat}_' if noise_cat != 0 else fname_central
    fname_left    = f'{fname_left}noise_{noise_cat}_' if noise_cat != 0 else fname_left
    fname_right   = f'{fname_right}noise_{noise_cat}_' if noise_cat != 0 else fname_right

    # Add the label converter and use the extracted frame number
    fname_central = f'{fname_central}{frame_number:06d}.jpg' if converter_label is None else f'{fname_central}{converter_label}{frame_number:06d}.jpg'
    fname_left    = f'{fname_left}{frame_number:06d}.jpg' if converter_label is None else f'{fname_left}{converter_label}{frame_number:06d}.jpg'
    fname_right   = f'{fname_right}{frame_number:06d}.jpg' if converter_label is None else f'{fname_right}{converter_label}{frame_number:06d}.jpg'

    # Save the masks, they are 2D numpy arrays, so we can use PIL
    Image.fromarray(mask_merge_central).save(os.path.join(base_path, route, fname_central))
    Image.fromarray(mask_merge_left).save(os.path.join(base_path, route, fname_left))
    Image.fromarray(mask_merge_right).save(os.path.join(base_path, route, fname_right))


def process_container(args) -> type(None):
    """
    Fix the can_bus files, producing the "cmd_fix_can_bus" files. In essence, we group the
    gyroscope and accelerometer data, as well as give the command/direction that the ego
    vehicle should take at the next intersection earlier than usual.
    """
    container_path, dataset_path = args
    container = container_path.split(os.sep)[-1]

    json_path_list = glob(os.path.join(container_path, 'can_bus*.json'))
    utils.sort_nicely(json_path_list)
    command_list=[]
    dist=[]
    for json_file in json_path_list:
        try:
            with open(json_file, 'r') as json_:
                data = json.load(json_)
                command = data['direction']
                command_list.append(command)
                dist.append(max(data['speed'], 0.0)* 0.1)

                # If accelerometer and gyro were saved differently, join them
                # That is, we have 'accelerometer_x': 0.0, 'accelerometer_y': 0.0, 'accelerometer_z': 0.0
                # and 'gyroscope_x': 0.0, 'gyroscope_y': 0.0, 'gyroscope_z': 0.0.
                # Replace these with "imu_acc": [0.0, 0.0, 0.0] and "imu_gyroscope": [0.0, 0.0, 0.0]
                if 'accelerometer_x' in data:
                    data['imu_acc'] = [data.pop('accelerometer_x'), 
                                    data.pop('accelerometer_y'), 
                                    data.pop('accelerometer_z')]
                if 'gyroscope_x' in data:
                    data['imu_gyroscope'] = [data.pop('gyroscope_x'), 
                                            data.pop('gyroscope_y'), 
                                            data.pop('gyroscope_z')]
            # Save the file with the new data
            with open(json_file, 'w') as fd:
                json.dump(data, fd, indent=4, sort_keys=True)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    latest_cmd = 4.0
    change_points=[]
    dist_list = []
    count_dist = 0.0
    cmd_value = [4.0]
    for idx, cmd in enumerate(command_list):
        if cmd != latest_cmd:
            cmd_value.append(cmd)
            change_points.append(idx)
            dist_list.append(count_dist)
            count_dist=0.0
            latest_cmd = cmd
        else:
            count_dist += dist[idx]
    dist_list.append(count_dist)

    fix_id = []
    fix_dist = []
    fix_value = []
    for i, _ in enumerate(dist_list):
        if cmd_value[i] == 4.0:
            pass
        elif cmd_value[i] == 5.0 or cmd_value[i] == 6.0:
            if cmd_value[i-1] == 6.0 or cmd_value[i-1] == 5.0:
                pass
            else:
                fix_id.append(change_points[i-1])
                fix_dist.append(float(min(math.floor(dist_list[i-1]), 6.0)))
                fix_value.append(cmd_value[i])
        elif cmd_value[i] == 1.0 or cmd_value[i] == 2.0 or cmd_value[i] == 3.0:
            if cmd_value[i-1] == 5.0 or cmd_value[i-1] == 6.0:
                pass
            else:
                if dist_list[i-1] < 6.0:
                    pass
                else:
                    fix_id.append(change_points[i - 1])
                    fix_dist.append(float(min(math.floor(dist_list[i-1]), 6.0)))
                    fix_value.append(cmd_value[i])

    files_to_be_fixed=[]
    values=[]
    for i, sample_id in enumerate(fix_id):
        count_dist=0.0
        last_id = (0 if i == 0 else fix_id[i-1])
        for frame_id, json_file in reversed(list(enumerate(json_path_list[last_id:sample_id]))):
            if count_dist < fix_dist[i]:
                with open(json_file) as json_:
                    data = json.load(json_)
                    speed = max(data['speed'], 0.0)
                    count_dist += speed * 0.1
                    files_to_be_fixed.append(json_file)
                    values.append(fix_value[i])

    for json_file in json_path_list:
        if json_file in files_to_be_fixed:
            with open(json_file) as json_:
                data = json.load(json_)
                pseudo_data=data
                pseudo_data['direction'] = values[files_to_be_fixed.index(json_file)]

            with open(os.path.join(container_path, f'cmd_fix_{json_file.split(os.sep)[-1]}'), 'w') as fd:
                json.dump(pseudo_data, fd, indent=4, sort_keys=True)
        else:
            shutil.copy(json_file, os.path.join(container_path, 'cmd_fix_' + json_file.split('/')[-1]))


def parse_tick_ranges(range_str: str) -> Union[int, List[Tuple[int, int]]]:
    """ Parse a string of tick ranges into a list of tuples. """
    if '-' not in range_str:
        return int(range_str)
    ranges = []
    for part in range_str.split(','):
        start, end = map(int, part.split('-'))
        ranges.append((start, end))
    return ranges


def is_tick_in_ranges(tick: int, 
                      ranges: Union[int, List[Tuple[int, int]]]) -> bool:
    """ Check if a tick is within any of the given ranges. If ranges is an int, 
        check if tick is greater or equal than it for elimination."""
    if isinstance(ranges, int):
        return tick >= ranges
    return any(start <= tick <= end for start, end in ranges)


def find_files(directory: Union[str, os.PathLike], 
               ranges: List[Tuple[int, int]]) -> List[str]:
    """ Find all files in the directory with ticks within the given ranges. """
    pattern = r'(\d+)(?=\.\w+$)'  # Regex pattern to extract the number before the file extension
    selected_files = []
    # Iterate over all files in the directory
    for entry in os.scandir(directory):
        if entry.is_file():
            match = re.search(pattern, entry.name)
            if match:
                file_number = int(match.group(1))
                if is_tick_in_ranges(file_number, ranges):
                    selected_files.append(entry.path)
    return selected_files
 

def get_frame_info(directory: str) -> Tuple[Dict[int, List[str]], int]:
    """Get information about frames and their files in a directory.
    Returns a dictionary of frame numbers to file paths and the expected file count per frame.
    """
    # Find all files with 6 digits in their name
    files = glob(os.path.join(directory, '**', '*[0-9][0-9][0-9][0-9][0-9][0-9].*'), recursive=True)
    
    # Group files by frame number
    frame_files = defaultdict(list)
    pattern = re.compile(r'.*?(\d{6})\.')
    
    for file in files:
        match = pattern.match(file)
        if match:
            frame_num = int(match.group(1))
            frame_files[frame_num].append(file)
    
    # Get the most common file count (this is our expected count per frame)
    if frame_files:
        counts = [len(files) for files in frame_files.values()]
        expected_count = max(set(counts), key=counts.count)
    else:
        expected_count = 0
    
    return frame_files, expected_count


def create_video_for_route(dataset_path, weather, route, fps, 
                           camera_name, output_path=None, json_filename: str = 'cmd_fix_can_bus'):
    def get_frame_number(filename):
        """Extract frame number from filename using regex."""
        match = re.search(r'(\d+)(?=\.\w+$)', filename)
        return int(match.group(1)) if match else None
    # Command to string
    command_sign_dict = {
                1.0: 'Turn Left',
                2.0: 'Turn Right',
                3.0: 'Go Straight',
                4.0: 'Follow Lane',
                5.0: 'Change Lane Left',
                6.0: 'Change Lane Right'
            }
    # Get the sensor data paths
    paths = get_paths(data_root=os.path.join(dataset_path, weather, route), 
                      sensors=[camera_name, json_filename])
    assert len(paths) % 4 == 0, f"Error, missing some data"

    left_images = {get_frame_number(path): path for path in paths if 'left' in path}
    central_images = {get_frame_number(path): path for path in paths if 'central' in path}
    right_images = {get_frame_number(path): path for path in paths if 'right' in path}
    can_bus = {get_frame_number(path): path for path in paths if json_filename in path}
    
    # Find the intersection of frame numbers that exist in all categories
    common_frames = set(left_images).intersection(central_images).intersection(right_images).intersection(can_bus)
    sorted_frames = sorted(common_frames)

    if not sorted_frames:
        print("No complete data sets available.")
        return

    # We will use the central camera as the reference for the video size
    central_img = cv2.imread(central_images[sorted_frames[-1]])

    height, width, _ = central_img.shape

    # Setup the video writer
    output_path = os.path.join(dataset_path, 'videos') if output_path is None else output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    video_name = os.path.join(output_path, f'{route}_{camera_name}.mp4')
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (3 * width, height))

    # Calculate font scale based on image width
    font_scale = width / 300.0  # Scale relative to original 300px width
    thickness = max(1, int(2 * font_scale))  # Scale thickness proportionally
    
    # Text position scaling
    x_margin = int(10 * font_scale)
    y_top = int(30 * font_scale)
    y_bottom = height - int(30 * font_scale)

    # Create the videos by horizontally concatenating the 3 cameras
    for frame_number in sorted_frames:
        left_img = cv2.imread(left_images[frame_number])
        central_img = cv2.imread(central_images[frame_number])
        right_img = cv2.imread(right_images[frame_number])

        # Get data from can bus
        with open(can_bus[frame_number]) as json_: 
            data = json.load(json_)
            speed = data['speed']
            steering = data['steer']
            acceleration = data['acceleration']
            command = command_sign_dict[data['direction']]
            position = data['ego_location']

        # Draw scaled text
        cv2.putText(left_img, f'Frame: {frame_number:06d}', 
                    (x_margin, y_top), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 0, 255), thickness)
                    
        cv2.putText(central_img, f'{command}', 
                    (x_margin, y_top), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 0, 0), thickness)
                    
        cv2.putText(right_img, f'Speed: {speed:.2f} m/s', 
                    (x_margin, y_top), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 0, 0), thickness)
                    
        cv2.putText(left_img, f'Steering: {steering:.2f}', 
                    (x_margin, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 255, 255), thickness)

        cv2.putText(central_img, f'Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})',
                    (x_margin, y_bottom), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale/2, (0, 255, 255), thickness)
                    
        cv2.putText(right_img, f'Acceleration: {acceleration:.2f}', 
                    (x_margin, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 255, 255), thickness)

        concat_img = cv2.hconcat([left_img, central_img, right_img])
        video.write(concat_img)
        
    video.release()
    print(f"Video for {weather}/{route} created successfully.")


def resize_image(args: tuple) -> None:
    """ 
    Resize an image to the given target size at the original directory.
    Uses different interpolation methods based on image type.
    """
    image_path, target_size, resized_img_prefix = args
    
    # Determine if this is a virtual attention map (grayscale)
    is_attention_map = 'virtual_attention' in image_path
    
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Choose interpolation method based on image type
    interpolation = cv2.INTER_AREA if is_attention_map else cv2.INTER_LINEAR
    
    # Resize the image
    resized_img = cv2.resize(img, target_size, interpolation=interpolation)
    
    # Extracting the directory and filename
    directory = os.path.dirname(image_path)
    filename = f'{resized_img_prefix}_{os.path.basename(image_path)}'
    
    # Saving the image in the same directory with the new filename
    cv2.imwrite(os.path.join(directory, filename), resized_img)


label2rgb_cityscapes = {
    'None': [0, 0, 0],
    'Building': [70, 70, 70],
    'Fence': [100, 40, 40],
    'Other': [55, 90, 80],
    'Pedestrian': [220, 20, 60],
    'Pole': [153, 153, 153],
    'Road Lines': [157, 234, 50],
    'Road': [128, 64, 128],
    'Sidewalk': [244, 35, 232],
    'Vegetation': [107, 142, 35],
    'Vehicle': [0, 0, 142],
    'Wall': [102, 102, 156],
    'Traffic Sign': [220, 220, 0],
    'Sky': [70, 130, 180],
    'Ground': [81, 0, 81],
    'Bridge': [150, 100, 100],
    'Rail Track': [230, 150, 140],
    'Guard Rail': [180, 165, 180],
    'Traffic Light': [250, 170, 30],
    'Statics': [110, 190, 160],
    'Dynamics': [170, 120, 50],
    'Water': [45, 60, 150],
    'Terrain': [145, 170, 100],
    'Curb': [255, 255, 100]
}

id2label_mapillary = {
    0: 'Bird',
    1: 'Ground Animal',
    2: 'Curb',
    3: 'Fence',
    4: 'Guard Rail',
    5: 'Barrier',
    6: 'Wall',
    7: 'Bike Lane',
    8: 'Crosswalk - Plain',
    9: 'Curb Cut',
    10: 'Parking',
    11: 'Pedestrian Area',
    12: 'Rail Track',
    13: 'Road',
    14: 'Service Lane',
    15: 'Sidewalk',
    16: 'Bridge',
    17: 'Building',
    18: 'Tunnel',
    19: 'Person',
    20: 'Bicyclist',
    21: 'Motorcyclist',
    22: 'Other Rider',
    23: 'Lane Marking - Crosswalk',
    24: 'Lane Marking - General',
    25: 'Mountain',
    26: 'Sand',
    27: 'Sky',
    28: 'Snow',
    29: 'Terrain',
    30: 'Vegetation',
    31: 'Water',
    32: 'Banner',
    33: 'Bench',
    34: 'Bike Rack',
    35: 'Billboard',
    36: 'Catch Basin',
    37: 'CCTV Camera',
    38: 'Fire Hydrant',
    39: 'Junction Box',
    40: 'Mailbox',
    41: 'Manhole',
    42: 'Phone Booth',
    43: 'Pothole',
    44: 'Street Light',
    45: 'Pole',
    46: 'Traffic Sign Frame',
    47: 'Utility Pole',
    48: 'Traffic Light',
    49: 'Traffic Sign (Back)',
    50: 'Traffic Sign (Front)',
    51: 'Trash Can',
    52: 'Bicycle',
    53: 'Boat',
    54: 'Bus',
    55: 'Car',
    56: 'Caravan',
    57: 'Motorcycle',
    58: 'On Rails',
    59: 'Other Vehicle',
    60: 'Trailer',
    61: 'Truck',
    62: 'Wheeled Slow',
    63: 'Car Mount',
    64: 'Ego Vehicle'}

# Mapillary label to CityScapes label dictionary (assuming best matches)
mapillary_to_cityscapes = {
    'Bird': 'Other',
    'Ground Animal': 'Other',
    'Curb': 'Curb',  # Keep Curb as its own class
    'Fence': 'Fence',
    'Guard Rail': 'Guard Rail',
    'Barrier': 'Fence',
    'Wall': 'Wall',
    'Bike Lane': 'Road Lines',
    'Crosswalk - Plain': 'Road Lines',
    'Curb Cut': 'Sidewalk',
    'Parking': 'Road',
    'Pedestrian Area': 'Sidewalk',
    'Rail Track': 'Rail Track',
    'Road': 'Road',
    'Service Lane': 'Road',
    'Sidewalk': 'Sidewalk',
    'Bridge': 'Bridge',
    'Building': 'Building',
    'Tunnel': 'Building',
    'Person': 'Pedestrian',
    'Bicyclist': 'Pedestrian',
    'Motorcyclist': 'Pedestrian',
    'Other Rider': 'Pedestrian',
    'Lane Marking - Crosswalk': 'Road Lines',
    'Lane Marking - General': 'Road Lines',
    'Mountain': 'Terrain',
    'Sand': 'Ground',
    'Sky': 'Sky',
    'Snow': 'Ground',
    'Terrain': 'Terrain',
    'Vegetation': 'Vegetation',
    'Water': 'Water',
    'Banner': 'Other',
    'Bench': 'Other',
    'Bike Rack': 'Other',
    'Billboard': 'Other',
    'Catch Basin': 'Other',
    'CCTV Camera': 'Other',
    'Fire Hydrant': 'Other',
    'Junction Box': 'Other',
    'Mailbox': 'Other',
    'Manhole': 'Other',
    'Phone Booth': 'Other',
    'Pothole': 'Other',
    'Street Light': 'Pole',
    'Pole': 'Pole',
    'Traffic Sign Frame': 'Traffic Sign',
    'Utility Pole': 'Pole',
    'Traffic Light': 'Traffic Light',
    'Traffic Sign (Back)': 'Traffic Sign',
    'Traffic Sign (Front)': 'Traffic Sign',
    'Trash Can': 'Other',
    'Bicycle': 'Vehicle',
    'Boat': 'Vehicle',
    'Bus': 'Vehicle',
    'Car': 'Vehicle',
    'Caravan': 'Vehicle',
    'Motorcycle': 'Vehicle',
    'On Rails': 'Vehicle',
    'Other Vehicle': 'Vehicle',
    'Trailer': 'Vehicle',
    'Truck': 'Vehicle',
    'Wheeled Slow': 'Vehicle',
    'Car Mount': 'Vehicle',
    'Ego Vehicle': 'Vehicle'
}

# Function to predict semantic segmentation (mock function for demonstration)
def predict_segmentation(image_path: str,
                         processor: image_processing_mask2former.Mask2FormerImageProcessor,
                         model: modeling_mask2former.Mask2FormerForUniversalSegmentation,
                         device: str = 'cuda') -> np.ndarray:
    # Mock prediction function: Replace with actual model prediction
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # you can pass them to processor for postprocessing
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    return predicted_semantic_map.detach().cpu().numpy()


def mapillary_to_cityscapes_rgb(segmentation: np.ndarray) -> np.ndarray:
    """
    Take a predicted segmentation and return an RGB image with CityScapes palette`.
    """
    h, w = segmentation.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for mapillary_id, label in id2label_mapillary.items():
        cityscapes_label = mapillary_to_cityscapes.get(label, 'None')
        rgb = label2rgb_cityscapes[cityscapes_label]
        rgb_image[segmentation == mapillary_id] = rgb

    return rgb_image

def process_image(image_path: str,
                  processor: image_processing_mask2former.Mask2FormerImageProcessor,
                  model: modeling_mask2former.Mask2FormerForUniversalSegmentation,
                  save_name_start: str = 'ss_hat',  # Predicted semantic segm.
                  extension: str = None,
                  device: str = 'cuda') -> None:
    # Predict segmentation
    segmentation = predict_segmentation(image_path, processor, model, device)
    
    # Convert to RGB image
    rgb_image = mapillary_to_cityscapes_rgb(segmentation)
    
    # Determine the output filename
    dir_name, base_name = os.path.split(image_path)
    name, ext = os.path.splitext(base_name)
    ext = extension if extension is not None else ext  # Extension override
    output_name = f"{save_name_start}_{name.split('_')[-1]}{ext}"
    output_path = os.path.join(dir_name, output_name)
    
    # Save the RGB image
    img = Image.fromarray(rgb_image)
    img.save(output_path)


def get_files_in_directory(root, prefixes):
    """Get files in a single directory that start with given prefixes."""
    files = []
    for f in os.listdir(root):
        if any(f.startswith(prefix) for prefix in prefixes):
            files.append(os.path.join(root, f))
    return files


def get_all_directories(directory):
    """Get all directories in the given directory, recursively."""
    all_dirs = []
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            all_dirs.append(os.path.join(root, d))
    return all_dirs


def get_files_with_prefix(directory, prefixes, num_workers=8):
    """Get all files in a directory and its subdirectories that start with given prefixes."""
    all_dirs = get_all_directories(directory)
    all_dirs.append(directory)  # Include the root directory itself

    file_paths = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_dir = {executor.submit(get_files_in_directory, dir_path, prefixes): dir_path for dir_path in all_dirs}
        for future in tqdm(as_completed(future_to_dir), total=len(future_to_dir), desc="Scanning directories", dynamic_ncols=True):
            dir_files = future.result()
            file_paths.extend(dir_files)

    return file_paths


def get_files_with_prefix_and_suffix(directory: str, prefixes: List[str], suffixes: List[str], num_workers: int = 8):
    """Get all files in a directory and its subdirectories that start with given prefixes and end with given suffixes."""
    all_dirs = get_all_directories(directory)
    all_dirs.append(directory)  # Include the root directory itself
    file_paths = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_dir = {executor.submit(get_files_in_directory, dir_path, prefixes): dir_path for dir_path in all_dirs}
        for future in tqdm(as_completed(future_to_dir), total=len(future_to_dir), desc="Scanning directories", dynamic_ncols=True):
            dir_files = future.result()
            file_paths.extend([f for f in dir_files if any(f.endswith(suffix) for suffix in suffixes)])
    return file_paths


def apply_mask_to_image(segmentation, mask):
    """Apply a mask to a segmentation image."""
    # Convert both images to numpy arrays
    seg_array = np.array(segmentation)
    mask_array = np.array(mask)[:,:, np.newaxis]  # Ensure mask has the same number of channels
    
    # Apply the mask (inverted)
    masked_image = seg_array * (mask_array == 0)
    
    return Image.fromarray(masked_image)


# Define a custom dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs["image_path"] = image_path
        inputs["image_size"] = [*image.size]
        return inputs

def collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {key: [item[key] for item in batch] for key in keys}
    collated_batch["pixel_values"] = torch.cat([x for x in collated_batch["pixel_values"]], dim=0)
    return collated_batch

def process_batch(batch, processor, model, device):
    pixel_values = batch["pixel_values"].to(device)
    image_paths = batch["image_path"]
    image_sizes = batch["image_size"]

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    
    predicted_semantic_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=image_sizes)
    
    return predicted_semantic_maps, image_paths

def save_segmentation(predicted_semantic_maps, image_paths, save_name_start, extension, auto_replace_prefix: bool = True):
    for seg_map, image_path in zip(predicted_semantic_maps, image_paths):
        # Transform segmentation image to cityscapes palette
        segmentation = seg_map.cpu().numpy()
        rgb_image = mapillary_to_cityscapes_rgb(segmentation)
        
        # Get the path, name, and extension of the RGB image
        dir_name, base_name = os.path.split(image_path)
        name, ext = os.path.splitext(base_name)
        ext = extension if extension is not None else ext  # Extension override
        output_name = f"{save_name_start}_{name.split('_')[-1]}{ext}"
        output_path = os.path.join(dir_name, output_name)
        
        img = Image.fromarray(rgb_image)
        img.save(output_path)

def save_segmentation_threaded(predicted_semantic_maps, image_paths, save_name_start, extension, auto_replace_prefix, num_workers=8):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(save_segmentation, [seg_map], [image_path], save_name_start, extension, auto_replace_prefix)
            for seg_map, image_path in zip(predicted_semantic_maps, image_paths)
        ]
        for future in as_completed(futures):
            future.result()

# ============================================================
# Helper functions to average attention maps

CAMERAS = ['left', 'central', 'right']
ATTENTION_TYPES = ['', 'dynamic', 'traffic', 'static']

def get_frame_number(filename):
    """Extract frame number from filename using regex."""
    match = re.search(r'(\d+)(?=\.\w+$)', filename)
    return int(match.group(1)) if match else None

def average_frames(frames):
    """Average a list of frame arrays."""
    return np.mean(frames, axis=0).astype(np.uint8)

def get_files_for_camera(route_path, prefix, camera, attention_type):
    if attention_type:
        file_pattern = f'^{prefix}_{camera}_{attention_type}'
    else:
        file_pattern = f'^{prefix}_{camera}_(?!dynamic|traffic|static)'
    
    files = [f for f in os.listdir(route_path) if re.match(file_pattern, f) and f.endswith('.jpg')]
    utils.sort_nicely(files)
    return files

def process_frame(args):
    route_path, filename, fps, sec, frame_dict, sorted_frames = args
    current_frame = get_frame_number(filename)
    i = sorted_frames.index(current_frame)
    start = max(0, i - fps * sec)
    end = min(i + fps * sec, len(sorted_frames) - 1)
    
    frame_range = sorted_frames[start:end+1]
    frames_to_average = [np.array(Image.open(frame_dict[frame])) for frame in frame_range]
    
    averaged_frame = average_frames(frames_to_average)
    return current_frame, averaged_frame, filename

def process_camera(args):
    route_path, prefix, fps, sec, camera, attention_type = args
    files = get_files_for_camera(route_path, prefix, camera, attention_type)
    
    if not files:
        print(f"No files found for camera {camera} with attention type {attention_type} in {route_path}")
        return route_path, {}, camera, attention_type

    frame_dict = {get_frame_number(f): os.path.join(route_path, f) for f in files}
    sorted_frames = sorted(frame_dict.keys())
    
    frame_args = [(route_path, f, fps, sec, frame_dict, sorted_frames) for f in files]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_frame, frame_args))
    
    averaged_frames = {frame: (avg_frame, filename) for frame, avg_frame, filename in results}
    return route_path, averaged_frames, camera, attention_type

def save_averaged_frames(route_path, averaged_frames, output_prefix, sec, camera, attention_type):
    for frame_number, (averaged_frame, original_filename) in averaged_frames.items():
        if output_prefix is None:
            base_name = os.path.splitext(original_filename)[0]
            output_filename = f"avg_{sec}sec_{base_name}.jpg"
        else:
            output_filename = f"{output_prefix}_{camera}_{attention_type}_{frame_number:06d}.jpg"
        output_path = os.path.join(route_path, output_filename)
        Image.fromarray(averaged_frame).save(output_path)



# ====================== Main functions ======================


@click.group()
def main():
    pass


# ============================================================

@main.command(name='predict-ss', help='Predict the semantic segmentation of the RGB images in the given directory.')
@click.option('--dataset-path', default='carla', help='Dataset root to convert.', type=click.Path(exists=True))
@click.option('--rgb-prefix', default=['rgb'], help='Prefixes of the RGB images to predict the semantic segmentation (e.g., rgb_central000124.png)', multiple=True)
@click.option('--rgb-extension', default='.png', help='Extension of the RGB data in case there are multiple versions (e.g., backups)', type=str)
# Data saving options
@click.option('--save-name-prefix', 'save_name_prefix', default='ss_hat', help='Prefix for the saved semantic segmentation images.', type=str)
@click.option('--save-extension', 'save_name_extension', required=True, help='Image extension for the saved semantic segmentation images.', type=click.Choice(['.png', '.jpg', '.jpeg']))
@click.option('--auto-replace-prefix', 'auto_replace_prefix', default=True, help='Find and remove the part of the name before the first underscore', type=bool)
# Optional
@click.option('--device', default='cuda', help='Device to use for prediction.', type=click.Choice(['cuda', 'cpu']))
@click.option('--num-workers', default=8, help='Number of workers to use for parallel processing.', type=click.IntRange(min=1))
@click.option('--gpu-id', default=None, help='GPU ID to use for prediction, if using gpu.', type=click.IntRange(min=0))
@click.option('--batch-size', default=16, help='Batch size for prediction.', type=click.IntRange(min=1))
def predict_semantic_segmentation(dataset_path, rgb_prefix, rgb_extension, save_name_prefix, save_name_extension, auto_replace_prefix, device, num_workers, gpu_id, batch_size):
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else '0'
    device = 'cuda' if gpu_id is not None and torch.cuda.is_available() else device

    model_name = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    image_paths = get_files_with_prefix_and_suffix(dataset_path, list(rgb_prefix), suffixes=[rgb_extension])
    utils.sort_nicely(image_paths)
    total_images = len(image_paths)

    dataset = ImageDataset(image_paths, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=3)
    
    with tqdm(total=total_images, desc="Processing images", unit="images", dynamic_ncols=True) as pbar:
        for batch in dataloader:
            predicted_semantic_maps, image_paths = process_batch(batch, processor, model, device)
            save_segmentation_threaded(predicted_semantic_maps, image_paths, save_name_prefix, save_name_extension, auto_replace_prefix, num_workers)
            pbar.update(len(image_paths))

    print('Done!')


@main.command(name='visualize-routes')
@click.option('--dataset-path', default='carla', help='Dataset root to visualize.', type=click.Path(exists=True), required=True)
@click.option('--fps', default=10.0, help='FPS of the video.', type=click.FloatRange(min=1.0), show_default=True)
@click.option('--camera-name', default='rgb', help='String prefix in the camera/sensor name to use for the video', required=True)
@click.option('--json-filename', default='cmd_fix_can_bus', help='Filename of the JSON file containing the data.', type=click.Choice(['can_bus', 'cmd_fix_can_bus']), show_default=True)
@click.option('--out', 'output_path', help='Output path for the videos. If not specified/None, will be in the directory of the dataset.', default=None, show_default=True)
# Additional params
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU.', type=click.IntRange(min=1))
def visualize_routes(dataset_path, fps, camera_name, json_filename, output_path: str = None, processes_per_cpu: int = 1) -> type(None):
    """ 
    Generate one video per route in the dataset. The structure of the dataset is as follows: 
        data_root/WEATHER/ROUTE/SENSOR_DATA  OR  data_root/ROUTE/SENSOR_DATA
    where WEATHER is one of the weather types (ClearNoon, etc.), ROUTE contains the route number,
    and SENSOR_DATA is the sensor data for that route, ordered in time, with the first tick number 
    starting in 00000. We will save the videos at the root of the dataset, in a subdirectory called 'videos'.
    """
    # Get all the weathers in the dataset
    weathers = sorted([weather for weather in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, weather))])
    weathers = [weather for weather in weathers if 'videos' not in weather]
    print('Weathers found: ', weathers)

    # Create a pool of worker processes
    num_cpus = os.cpu_count()
    pool = Pool(processes=num_cpus * processes_per_cpu)

    # Schedule the video creation tasks
    for weather in weathers:
        routes = sorted([route for route in os.listdir(os.path.join(dataset_path, weather)) if os.path.isdir(os.path.join(dataset_path, weather, route))])
        for route in routes:
            pool.apply_async(create_video_for_route, args=(dataset_path, weather, route, fps, camera_name, output_path, json_filename))
    pool.close()
    pool.join()

    print('Done!')


@main.command(name='prepare-ss')
@click.option('--dataset-path', default='carla', help='Dataset root to convert.', type=click.Path(exists=True))
@click.option('--ss-prefix', default='ss', help='Prefix of the semantic segmentation images', show_default=True)
# Additional params
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU.', type=click.IntRange(min=1))
@click.option('--debug', is_flag=True, help='Debug mode.')
def prepare_ss(dataset_path, ss_prefix: str = 'ss', processes_per_cpu: int = 1, debug: bool = False) -> type(None):
    """ Convert the dataset's semantic segmentation images to RGB if they are not (if only one channel has all the info) """
    # First, start with getting all the subdirectories in the dataset; usual structure: data_root/subroute/route_00001/rgb_central06d.jpg, for example
    subdatasets = sorted([subdir for subdir in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subdir))])
    print('Subdatasets found: ', subdatasets) if debug else None

    with Pool(os.cpu_count() * processes_per_cpu) as pool:
        for subdata in subdatasets:
            # Get the routes in the subdataset
            routes = sorted([route for route in os.listdir(os.path.join(dataset_path, subdata)) if os.path.isdir(os.path.join(dataset_path, subdata, route))])
            print('Routes found: ', routes) if debug else None

            for route in routes:
                # Get the sensor data paths
                paths = get_paths(data_root=os.path.join(dataset_path, subdata, route), sensors=[ss_prefix])
                
                # Let's get the paths for the 3 cameras of semantic segmentation
                semantic_segmentation_paths = [path for path in paths if os.path.basename(path).startswith(ss_prefix)]

                args = [(path, dataset_path, subdata, route) for path in semantic_segmentation_paths]
                for _ in tqdm(pool.imap(prepare_semantic_segmentation, args), total=len(args), 
                              dynamic_ncols=True, desc=f'Preparing the semantic segmentation images [{subdata}/{route}]'):
                    pass

    print('Done!')


class InfiniteNone:
    def __getitem__(self, index):
        return None
    
    def __len__(self):
        return float('inf')
    
def process_route(pool, base_path, route, sensor_names, ignore_depth, depth_threshold, min_depth, noise_cat, converter_label: str = None):
    # Get the sensor data paths
    paths = get_paths(data_root=os.path.join(base_path, route), sensors=sensor_names)

    # Let's get the paths for the 3 cameras of depth and ss, as well as the can bus
    depth_paths = [path for path in paths if os.path.basename(path).startswith('depth_')] if not ignore_depth else InfiniteNone()
    semantic_segmentation_paths = [path for path in paths if os.path.basename(path).startswith('ss_')]
    # can_bus_paths = [path for path in paths if 'can_bus' in path.split(os.sep)[-1]]

    if not ignore_depth:
        assert len(depth_paths) == len(semantic_segmentation_paths), \
            f"Error, sensor mismatch: number of Depth paths: {len(depth_paths)}, SS paths: {len(semantic_segmentation_paths)}"
    else:
        pass

    num_data_route = len(semantic_segmentation_paths) // 3

    # Prepare the semantic segmentation images before
    args = [(idx, noise_cat, depth_paths, semantic_segmentation_paths, depth_threshold, min_depth,
             num_data_route, base_path, route, converter_label) for idx in range(num_data_route)]
    for _ in tqdm(pool.imap(process_map, args), total=num_data_route, dynamic_ncols=True,
                  desc=f'Generating the virtual attention maps [{os.path.basename(base_path)}/{route}]'):
        pass



@main.command(name='create-virtual-attentions')
@click.option('--dataset-path', default='carla', help='Dataset root to convert.', type=click.Path(exists=True))
@click.option('--ignore-depth', is_flag=True, help='Ignore the depth images and only use the semantic segmentation images.')
@click.option('--ss-prefix', 'ss_name', default='ss', help='Prefix string of the semantic segmentation images.', type=str)
@click.option('--max-depth', 'depth_threshold', default=100.0, help='Filter out objects beyond this depth.', type=click.FloatRange(min=0.0), show_default=True)
@click.option('--min-depth', 'min_depth', default=1.7, help='Filter out objects starting from this depth for the central camera. Default takes into account the hood of the car, if shown in the central camera.', type=click.FloatRange(min=0.0), show_default=True)
# Virtual attention maps options
@click.option('--converter-label', 'converter_label', default=None, help='Label to convert the semantic segmentation to. If not provided, will use the default class selection.', 
                type=click.Choice(['pedestrian', 'vehicle', 'trafficlight', 'trafficsign', 'lane', 'pole', 'pedestrian-lane', 'vehicle-lane', 'trafficlight-lane', 'trafficsign-lane', 'pole-lane', 'dynamic', 'traffic', 'static']))
@click.option('--noise-cat', 'noise_cat', default=0, help='Noise category to use for the virtual attention maps; Perlin Noise (PN) and Grid Perlin Noise (GPN). 0: No noise; 1: (global) GPN; 2: GPN on objects and PN on lines; 3: (global) PN', type=click.IntRange(min=0, max=3), show_default=True)
@click.option('--seed', 'seed', default=None, help='Seed for the noise generation.', type=click.INT)
# Additional params
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU core.', type=click.IntRange(min=1))
@click.option('--debug', is_flag=True, help='Debug mode.')
def create_virtual_atts(dataset_path: Union[str, os.PathLike], ignore_depth: bool, ss_name: str, 
                        depth_threshold: float, min_depth: float, converter_label: str, noise_cat: int, 
                        seed: int, processes_per_cpu: int = 1, debug: bool = False) -> Type[None]:
    """ Generate the virtual attention maps for the dataset using the depth and semantic segmentation images. """
    print(f'Creating virtual attention maps for label "{converter_label}"') if converter_label is not None else None
    # Set the seed for the noise generation, if specified
    if seed is not None:
        from _utils import training_utils
        training_utils.seed_everything(seed)
    # First, start with getting all the subdirectories in the dataset; usual structure: data_root/subroute/route_00001/rgb_central06d.jpg, for example
    subdatasets = sorted([subdir for subdir in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subdir))])
    print('Subdatasets found: ', subdatasets) if debug else None

    sensor_names = ['cmd_fix_can_bus', ss_name] if ignore_depth else ['cmd_fix_can_bus', 'depth', ss_name]

    with Pool(os.cpu_count() * processes_per_cpu) as pool:
        if subdatasets:
            # Case 1 and Case 3: Process subdirectories, which may contain further directories (routes)
            for subdir in subdatasets:
                subdir_path = os.path.join(dataset_path, subdir)
                routes = sorted([route for route in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, route))])
                
                if routes:
                    print(f'Routes found in subdirectory {subdir}:', routes) if debug else None
                    for route in routes:
                        process_route(pool, subdir_path, route, sensor_names, ignore_depth, depth_threshold, min_depth, noise_cat, converter_label)
                else:
                    print(f'Treating subdirectory {subdir} as a route') if debug else None
                    process_route(pool, dataset_path, subdir, sensor_names, ignore_depth, depth_threshold, min_depth, noise_cat, converter_label)
        else:

            # Case 2: No subdirectories found, routes are directly under dataset_path
            routes = sorted([route for route in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, route))])
            print('Routes found at dataset root:', routes) if debug else None

            for route in routes:
                process_route(pool, dataset_path, route, sensor_names, ignore_depth, depth_threshold, min_depth, noise_cat, converter_label)

    print('Done!')


@main.command(name='command-fix')
@click.option('--dataset-path', help='Path to the root of your dataset to modify', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
def command_fix(dataset_path: Union[str, os.PathLike]):
    """ Manually fix a bug in the dataset wherein the command/direction is given too soon to the ego vehicle. """
    all_containers_path_list = find_deepest_directories(dataset_path)
    # all_containers_path_list = glob(os.path.join(dataset_path, '*'))
    all_containers_path_list = [path for path in all_containers_path_list if 'videos' not in path]
    utils.sort_nicely(all_containers_path_list)

    args = [(container_path, dataset_path) for container_path in all_containers_path_list]

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap(process_container, args), total=len(all_containers_path_list), dynamic_ncols=True):
            pass


@main.command(name='clean-route')
@click.option('--route-path', help='Path to the root of your route to modify', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
@click.option('--remove-ticks', 
              help='Either ranges of ticks to remove (e.g., 800-900 will remove ticks 800 through 900 inclusive) '
                   'or a single number (e.g., 99 will remove tick 99 and all ticks after it)', 
              required=True)
@click.option('--clean-type', type=click.Choice(['delete', 'move']), 
              help='Whether to delete or move the files', required=True)
@click.option('--invalid-path', 
              help='Path to move invalid files to. Required when --clean-type=move',
              type=click.Path(file_okay=False, dir_okay=True))
@click.option('--num-parent-dirs', 
              help='Number of parent directories to preserve in the invalid path structure',
              type=click.INT, default=2)
def clean_route(route_path: Union[str, os.PathLike], remove_ticks: str, clean_type: str,
                invalid_path: Union[str, os.PathLike, None], num_parent_dirs: int = 2):
    """Remove or move all files containing ticks within the specified ranges in their file name.
    Also handles frames that don't have the expected number of files."""
    
    # Validate parameters
    if clean_type == 'move' and not invalid_path:
        raise click.UsageError("--invalid-path is required when --clean-type=move")
    elif clean_type == 'delete' and invalid_path:
        raise click.UsageError("--invalid-path should not be provided when --clean-type=delete")

    # Get frame information
    frame_files, expected_count = get_frame_info(route_path)
    
    if not frame_files:
        print(f'No files found in {route_path}')
        return
    
    # Parse tick ranges
    tick_ranges = parse_tick_ranges(remove_ticks)
    
    # Find files to clean
    files_to_clean = set()
    mismatched_frames = []  # Keep track of frames with wrong file count
    range_frames = []       # Keep track of frames in specified ranges
    
    for frame_num, files in frame_files.items():
        # Check for mismatched file count
        if len(files) != expected_count:
            files_to_clean.update(files)
            mismatched_frames.append(frame_num)
            
        # Check if frame is in specified ranges
        if is_tick_in_ranges(frame_num, tick_ranges):
            files_to_clean.update(files)
            range_frames.append(frame_num)

    if not files_to_clean:
        print(f'No files found matching the criteria in {route_path}')
        return

    # Print diagnostic information
    print("\nDiagnostic Information:")
    print(f"Expected files per frame: {expected_count}")
    
    if mismatched_frames:
        print(f"\nFound {len(mismatched_frames)} frames with incorrect file count:")
        for frame_num in sorted(mismatched_frames):
            files = [os.path.basename(f) for f in frame_files[frame_num]]
            print(f"\nFrame {frame_num} ({len(files)} files instead of {expected_count}):")
            for f in sorted(files):
                print(f"  - {f}")
    
    if range_frames:
        print(f"\nFound {len(range_frames)} frames in specified ranges:")
        print(f"Range frames: {sorted(range_frames)[:5]}{'...' if len(range_frames) > 5 else ''}")

    # Prepare message based on operation type
    message = f'\nAre you sure you want to {clean_type} {len(files_to_clean)} files from {route_path}?'
    if clean_type == 'move':
        message += f'\nFiles will be moved to {invalid_path} maintaining the directory structure.'
    else:
        message += '\nWARNING: This operation is irreversible!'
    
    print(message)
    print('Type "yes" to confirm, anything else to cancel.')
    user_input = input()
    if user_input != 'yes':
        print('Aborting...')
        return

    if clean_type == 'move':
        # Create the invalid path directory if it doesn't exist
        os.makedirs(invalid_path, exist_ok=True)

        # Get the part of the path we want to preserve
        path_components = os.path.normpath(route_path).split(os.sep)
        preserved_path = os.sep.join(path_components[-num_parent_dirs:])
        dest_base_dir = os.path.join(invalid_path, preserved_path)
        os.makedirs(dest_base_dir, exist_ok=True)

        # Move each file
        for src_file in tqdm(files_to_clean, desc=f'Moving files to {dest_base_dir}', dynamic_ncols=True):
            filename = os.path.basename(src_file)
            dest_file = os.path.join(dest_base_dir, filename)
            try:
                shutil.move(src_file, dest_file)
            except Exception as e:
                print(f'Error moving file {src_file} to {dest_file}: {e}')
                continue
    else:  # delete
        # Delete the files
        for file in tqdm(files_to_clean, desc='Deleting files', dynamic_ncols=True):
            try:
                os.remove(file)
            except Exception as e:
                print(f'Error deleting file {file}: {e}')
                continue

    print(f'{clean_type.capitalize()} operation completed successfully!')


@main.command(name='resize-dataset')
@click.option('--dataset-path', help='Path to the root of your dataset to modify', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
@click.option('--res', 'target_resolution', help='Resolution to resize the images to; give it in format WxH, e.g. 100x50.', type=click.STRING, required=True)
@click.option('--resized-prefix', 'resized_img_prefix', help='Prefix to add to the resized image names', type=click.STRING, default='resized', show_default=True)
@click.option('--img-ext', 'ext', default='jpg', help='Image extension to look for.', type=click.STRING, show_default=True)
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU.', type=click.IntRange(min=1))
@click.option('--img-prefixes', help='Comma-separated list of image prefixes to resize (e.g., "rgb,depth,ss")', type=click.STRING, default='rgb', show_default=True)
def resize_dataset(dataset_path: Union[str, os.PathLike], target_resolution: str, resized_img_prefix: str = 'resized', ext: str = 'jpg', processes_per_cpu: int = 1, img_prefixes: str = 'rgb'):
    """
    Resize images in a dataset to a specified resolution. The resized images will 
    be saved in the same directory, with the specified prefix.
    
    Handles both regular RGB images and grayscale images (like virtual attention maps)
    with appropriate interpolation methods.
    """
    # Parse the image prefixes
    prefixes = [p.strip() for p in img_prefixes.split(',')]
    
    # Find all matching images for each prefix
    all_images = []
    for prefix in prefixes:
        images = glob(os.path.join(dataset_path, '**', '*', f'{prefix}*.{ext}'), recursive=True)
        all_images.extend(images)
    
    if not all_images:
        print(f"No images found with prefixes {prefixes} and extension .{ext}")
        return
    
    # Get the target size
    target_size = tuple(map(int, target_resolution.split('x')))
    print(f"Found {len(all_images)} images to resize")
    
    args = [(img, target_size, resized_img_prefix) for img in all_images]
    
    with Pool(os.cpu_count() * processes_per_cpu) as pool:
        for _ in tqdm(pool.imap(resize_image, args), total=len(all_images), 
                     desc='Resizing images', dynamic_ncols=True):
            pass



@main.command(name='average-virtual-attention')
@click.option('--dataset-path', type=click.Path(exists=True), help='Path to the dataset root')
@click.option('--prefix', type=str, default='virtual_attention', help='Prefix of the files to average')
@click.option('--fps', type=int, default=10, help='Frames per second of the dataset; intrinsic to how it was saved!')
@click.option('--sec', type=int, default=2, help='Seconds to consider for averaging.')
@click.option('--output-prefix', type=str, default=None, help='Prefix for output files. If None, will be "avg_{sec}sec_"')
@click.option('--num-workers', type=int, default=os.cpu_count(), help='Number of worker for multiprocessing. Default is number of CPUs.')
@click.option('--attention-type', type=click.Choice(ATTENTION_TYPES), default='', help='Type of virtual attention to process.')
def average_virtual_attention(dataset_path, prefix, fps, sec, output_prefix, num_workers, attention_type):
    routes = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    all_tasks = []
    for route in routes:
        for camera in CAMERAS:
            all_tasks.append((route, prefix, fps, sec, camera, attention_type))
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_camera, task) for task in all_tasks]
        
        for future in tqdm(as_completed(futures), total=len(all_tasks), desc="Processing routes and cameras"):
            route_path, averaged_frames, camera, att_type = future.result()
            if averaged_frames:
                save_averaged_frames(route_path, averaged_frames, output_prefix, sec, camera, att_type)

    print("Averaging complete!")

# TODO: why don't we use this function?
def extract_frame_number_flexible(filepath: str) -> int:
    """
    Extract frame number from filepath using multiple patterns.
    Handles both simple (name_000123.ext) and timestamp (name_timestamp_000123.ext) formats.
    """
    # Try different patterns for frame number extraction
    patterns = [
        r'(\d{6})(?=\.\w+$)',  # 6 digits before extension
        r'_(\d{6})(?=\.\w+$)', # underscore + 6 digits before extension
        r'(\d+)(?=\.\w+$)',    # any digits before extension
        r'_(\d+)_\d+\.', # frame number before timestamp
    ]
    
    for pattern in patterns:
        match = re.search(pattern, os.path.basename(filepath))
        if match:
            return int(match.group(1))
    
    return None

def group_files_by_sorted_order(file_paths: List[str], sensors_used: List[str]) -> Dict[int, Dict[str, str]]:
    """
    Group files by sorting them and assigning sequential frame numbers.
    Uses the sensors defined in config (e.g., g_conf.DATA_USED).
    
    Args:
        file_paths: List of all file paths
        sensors_used: List of sensor names from config (e.g., ['sekonix_60', 'sekonix_120', 'conti_front'])
    
    Returns: {frame_num: {'sekonix_60': path, 'sekonix_120': path, ...}}
    """
    # Group files by sensor type
    sensor_files = {sensor: [] for sensor in sensors_used}
    can_bus_files = []
    
    for filepath in file_paths:
        basename = os.path.basename(filepath)
        
        # Check for CAN bus data
        if 'can_bus' in basename or 'il_data' in basename:
            can_bus_files.append(filepath)
            continue
            
        # Check which sensor this file belongs to
        for sensor in sensors_used:
            if sensor in basename:
                sensor_files[sensor].append(filepath)
                break
    
    # Sort files for each sensor using natural sorting
    for sensor in sensors_used:
        utils.sort_nicely(sensor_files[sensor])  # This handles both numeric and timestamp sorting
    
    utils.sort_nicely(can_bus_files)
    
    # Find the minimum number of files across all sensors
    min_files = min(len(sensor_files[sensor]) for sensor in sensors_used if sensor_files[sensor])
    
    if can_bus_files:
        min_files = min(min_files, len(can_bus_files))
    
    # Group by frame number (index position after sorting)
    frame_groups = {}
    for frame_num in range(min_files):
        frame_data = {}
        
        # Add sensor data
        for sensor in sensors_used:
            if frame_num < len(sensor_files[sensor]):
                frame_data[sensor] = sensor_files[sensor][frame_num]
        
        # Add CAN bus data
        if frame_num < len(can_bus_files):
            frame_data['can_bus'] = can_bus_files[frame_num]
        
        # Only add frame if we have all required sensors
        if len(frame_data) == len(sensors_used) + (1 if can_bus_files else 0):
            frame_groups[frame_num] = frame_data
    
    return frame_groups


def batch_model_inference(frames_data, model, batch_size=8):
    """Batch process model inference for multiple frames."""
    all_results = []
    
    # Process in batches
    for i in tqdm(range(0, len(frames_data), batch_size), desc="Batch inference"):
        batch_frames = frames_data[i:i + batch_size]
        batch_data = []
        batch_gt_actions = []
        
        # Prepare batch data
        for frame_data in batch_frames:
            datapoint = {}
            datapoint['can_bus'] = dict()

            # Load CAN bus data
            with open(frame_data['can_bus'], 'r') as f:
                canbus_data = json.load(f)

            for value in g_conf.TARGETS + g_conf.OTHER_INPUTS:
                datapoint['can_bus'][value] = canbus_data[value]
            datapoint['can_bus'] = canbus_normalization(datapoint['can_bus'], g_conf.DATA_NORMALIZATION)
            
            # Save ground-truth action
            gt_action = [datapoint['can_bus']['steer'], datapoint['can_bus']['acceleration']]
            batch_gt_actions.append(gt_action)

            # Load images for each sensor in DATA_USED
            for sensor_type in g_conf.DATA_USED:
                if sensor_type in frame_data:
                    img = eval_utils.open_image(os.path.dirname(frame_data[sensor_type]), 
                                            os.path.basename(frame_data[sensor_type]))
                    datapoint[sensor_type] = img

            data = train_transform(datapoint, tuple(g_conf.IMAGE_SHAPE))
            batch_data.append(data)
        
        # Batch model inference
        with torch.no_grad():
            batch_results = []
            for data in batch_data:  # Still process individually due to model constraints
                action_output, resnet_inter, attn_weights, encoder_output = eval_utils.model_forward(
                    model, data, last_encoder_state=True
                )
                pred_action = action_output.squeeze().detach().cpu().numpy().tolist()
                batch_results.append({
                    'pred_action': pred_action,
                    'resnet_inter': resnet_inter,
                    'attn_weights': attn_weights,
                    'encoder_output': encoder_output,
                    'data': data
                })
        
        # Store results with GT actions
        for j, result in enumerate(batch_results):
            result['gt_action'] = batch_gt_actions[j]
            all_results.append(result)
    
    return all_results

def batch_model_inference_with_precompute(frames_data, model, batch_size=16):
    """Batch process with all visualization preprocessing."""
    all_results = []
    
    for i in tqdm(range(0, len(frames_data), batch_size), desc="Batch inference + preprocessing", dynamic_ncols=True):
        batch_frames = frames_data[i:i + batch_size]
        batch_data = []
        batch_gt_actions = []
        batch_positions = []
        
        # Prepare batch data
        for frame_data in batch_frames:
            datapoint = {}
            datapoint['can_bus'] = dict()

            # Load CAN bus data
            with open(frame_data['can_bus'], 'r') as f:
                canbus_data = json.load(f)
                                
            for value in g_conf.TARGETS + g_conf.OTHER_INPUTS:
                datapoint['can_bus'][value] = canbus_data[value]
            datapoint['can_bus'] = canbus_normalization(datapoint['can_bus'], g_conf.DATA_NORMALIZATION, shift_command=2)
            
            # Save ground-truth action
            gt_action = [datapoint['can_bus']['steer'], datapoint['can_bus']['acceleration']]
            batch_gt_actions.append(gt_action)
            
            # Save the ego position
            batch_positions.append(canbus_data.get('ego_position', None))

            # Load images for each sensor in DATA_USED
            for sensor_type in g_conf.DATA_USED:
                if sensor_type in frame_data:
                    img = eval_utils.open_image(os.path.dirname(frame_data[sensor_type]), 
                                            os.path.basename(frame_data[sensor_type]))
                    datapoint[sensor_type] = img

            data = train_transform(datapoint, tuple(g_conf.IMAGE_SHAPE))
            batch_data.append(data)
        
        
        # Batch model inference
        with torch.no_grad():
            batch_results = []
            for data in batch_data:
                action_output, resnet_inter, attn_weights, encoder_output = eval_utils.model_forward(
                    model, data, last_encoder_state=True
                )
                pred_action = action_output.squeeze().detach().cpu().numpy().tolist()
                
                # PRE-COMPUTE EXPENSIVE VISUALIZATIONS HERE
                processed_viz = precompute_visualizations(
                    data, resnet_inter, attn_weights, encoder_output, model, g_conf
                )
                
                batch_results.append({
                    'pred_action': pred_action,
                    'data': data,
                    'viz_data': processed_viz  # All expensive computations done
                })
        
        # Store results with GT actions
        for j, result in enumerate(batch_results):
            result['gt_action'] = batch_gt_actions[j]
            result['ego_position'] = batch_positions[j]
            all_results.append(result)
    
    return all_results

def precompute_visualizations(data, resnet_inter, attn_weights, encoder_output, model, config):
    """Pre-compute all expensive visualization operations."""
    src_images = [data[camera_type] for camera_type in config.DATA_USED 
                  if any(cam_type in camera_type for cam_type in ['rgb', 'sekonix', 'conti'])]
    
    # Pre-compute image concatenation and normalization
    img_cat = torch.cat(src_images[:3], dim=2)
    img_cat_norm = TF.normalize(img_cat, [-0.485/0.229, -0.456/0.224, -0.406/0.255], 
                          [1/0.229, 1/0.224, 1/0.255])
    img_cat_np = torch.clamp(img_cat_norm, 0, 1).cpu().numpy().transpose(1, 2, 0)
    # Change channels: RGB -> BGR for OpenCV compatibility if needed
    img_cat_np = img_cat_np[..., ::-1]
    
    # Pre-compute individual camera images
    camera_images = []
    for img in src_images[:3]:
        img_denorm = TF.normalize(img, [-0.485/0.229, -0.456/0.224, -0.406/0.255], 
                             [1/0.229, 1/0.224, 1/0.255])
        img_np = torch.clamp(img_denorm, 0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        camera_images.append(img_np)
    
    # Pre-compute ResNet features
    resnet_features = resnet_inter[-1]
    resnet_maps = []
    for i in range(min(3, len(src_images))):
        features = resnet_features[i]
        feature_map = torch.max(features, dim=0)[0].detach().cpu().numpy()
        resnet_maps.append(feature_map)
    
    # Pre-compute attention maps
    num_cameras = len(src_images[:3])
    start_idx = 0
    if model.num_register_tokens > 0:
        start_idx += model.num_register_tokens
    if not config.NO_ACT_TOKENS:
        start_idx += 2
    if config.CMD_SPD_TOKENS:
        start_idx += 2
    
    attention_maps = []
    target_size = (img_cat_np.shape[1], img_cat_np.shape[0])
    
    for attn in attn_weights:
        attn_avg = attn.squeeze().mean(dim=0)
        spatial_tokens = attn_avg[start_idx:]
        attn_reshaped = rearrange(
            spatial_tokens, 
            '(h w cam) -> 1 h (w cam)' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else '(cam h w) -> 1 h (cam w)',
            cam=num_cameras, h=model.res_out_h)
        
        attn_np = attn_reshaped.detach().cpu().numpy().transpose(1, 2, 0)
        attn_resized = cv2.resize(attn_np, target_size, interpolation=cv2.INTER_LINEAR)
        attention_maps.append(attn_resized)
    
    # Pre-compute L2 norm
    spatial_features = encoder_output.squeeze()[start_idx:]
    l2_norm = torch.norm(spatial_features, p=2, dim=1)
    
    # Calculate entropy
    l2_np = l2_norm.detach().cpu().numpy()
    counts, bins = np.histogram(l2_np, bins=50)
    probs = counts / np.sum(counts)
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    
    l2_reshaped = rearrange(l2_norm, 
                            '(h w cam) -> 1 h (w cam)' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else '(cam h w) -> 1 h (cam w)', 
                           cam=num_cameras, h=model.res_out_h)
    l2_np_reshaped = l2_reshaped.detach().cpu().numpy().transpose(1, 2, 0)
    l2_resized = cv2.resize(l2_np_reshaped, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Pre-compute PCA (most expensive operation)
    threshold = 0
    spatial_size, latent_dim = spatial_features.shape
    features = spatial_features.reshape(-1, latent_dim).detach().cpu().numpy()
    
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    
    bg_mask = pca_features[:, 0] < threshold
    fg_mask = ~bg_mask
    
    pca_features_fg = pca.transform(features[fg_mask])
    for i in range(3):
        pca_features_fg[:, i] = minmax_scale(pca_features_fg[:, i])
        
    pca_features_rgb = pca_features.copy()
    pca_features_rgb[bg_mask] = 0
    pca_features_rgb[fg_mask] = pca_features_fg
    
    pca_reshaped = rearrange(pca_features_rgb, 
                            '(h w cam) c -> c h (w cam) ' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else '(cam h w) c -> c h (cam w)',
                            cam=num_cameras, h=model.res_out_h, c=3)
    pca_np = pca_reshaped.transpose(1, 2, 0)
    pca_resized = cv2.resize(pca_np, target_size, interpolation=cv2.INTER_LINEAR)
    pca_resized = np.clip(pca_resized, 0, 1)
    
    # Add the direction command if available
    direction_onehot = data['can_bus']['direction']
    direction_str = decode_onehot_directions_to_str(direction_onehot)
    
    return {
        'camera_images': camera_images,
        'img_cat_np': img_cat_np,
        'resnet_maps': resnet_maps,
        'attention_maps': attention_maps,
        'l2_resized': l2_resized,
        'l2_range': (l2_norm.min().item(), l2_norm.max().item()),
        'l2_entropy': entropy,
        'pca_resized': pca_resized,
        'threshold': threshold,
        'direction_command': direction_str,
        'speed': data['can_bus']['speed'], # Normalized; need to denormalize if needed
    }


def create_opencv_visualization_frame(frame_num, action_history, prediction_history, position_history,
                                    config, viz_data) -> np.ndarray:
    """Create visualization using OpenCV - with proper colors and normalization."""
    
    # Define layout dimensions
    img_h, img_w = 252, 400  # Size for each "cell"
    rows, cols = 5, 6
    canvas_h, canvas_w = rows * img_h, cols * img_w
    
    # Create WHITE canvas (instead of black)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # Helper function to ensure uint8 and proper format
    def ensure_uint8_bgr(image, is_resnet=False):
        """Ensure image is uint8 and in BGR format for OpenCV."""
        if len(image.shape) == 3:
            # RGB image
            if image.dtype != np.uint8:
                if image.max() <= 1.0:  # Normalized float image
                    image = (image * 255).astype(np.uint8)
                else:  # Float image in 0-255 range
                    image = image.astype(np.uint8)
            # DON'T convert RGB to BGR here for camera images - they're already RGB
            # Only convert if this is a fresh RGB image
            if image.shape[2] == 3:  # Check if it needs conversion
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # Grayscale - apply proper normalization and colormap
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image, dtype=np.uint8)
            
            # Apply colormap - invert for attention/L2 norm, but NOT for ResNet
            if is_resnet:
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)  # No inversion
            else:
                colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_JET)  # Invert
            image = colored
        
        return image
    
    # Helper function to apply proper colormap to heatmaps
    def apply_heatmap_colormap(image, invert=True):
        """Apply colormap to heatmap data with proper normalization."""
        # Normalize to 0-255 range per frame
        if image.dtype != np.uint8:
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image, dtype=np.uint8)
        else:
            normalized = image
            
        # Apply colormap with optional inversion to match matplotlib
        if invert:
            colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_JET)
        else:
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
        return colored
    
    # Helper function to place image in grid
    def place_image(image, row, col, row_span=1, col_span=1, resize_mode=cv2.INTER_LINEAR, is_resnet=False):
        start_y, start_x = row * img_h, col * img_w
        end_y, end_x = start_y + row_span * img_h, start_x + col_span * img_w
        
        # Resize image to fit the cell(s)
        target_h, target_w = end_y - start_y, end_x - start_x
        
        # Ensure proper format and resize
        image_bgr = ensure_uint8_bgr(image, is_resnet)
        resized = cv2.resize(image_bgr, (target_w, target_h), resize_mode)
        
        canvas[start_y:end_y, start_x:end_x] = resized
        return start_y, start_x, end_y, end_x
    
    # Helper function to create overlay (with type safety)
    def create_overlay(base_img, overlay_img, base_alpha=0.4, overlay_alpha=0.6, invert_overlay=True):
        """Create overlay with proper type handling."""
        # Ensure base is uint8 BGR
        base_bgr = ensure_uint8_bgr(base_img)
        
        # Handle overlay with proper colormap
        if len(overlay_img.shape) == 2:  # Grayscale overlay
            overlay_colored = apply_heatmap_colormap(overlay_img, invert=invert_overlay)
        else:  # Already colored
            overlay_colored = ensure_uint8_bgr(overlay_img)
        
        # Ensure same dimensions
        if base_bgr.shape != overlay_colored.shape:
            overlay_colored = cv2.resize(overlay_colored, (base_bgr.shape[1], base_bgr.shape[0]))
        
        # Create weighted overlay
        result = cv2.addWeighted(base_bgr, base_alpha, overlay_colored, overlay_alpha, 0)
        return result
    
    # Helper function to add text (black text on white background)
    def add_text(text, x, y, scale=0.7, color=(0, 0, 0), thickness=2):  # Black text
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    
    # Row 1: RGB Images (3 cols) + Attention Layer 1 (3 cols)
    for i, img_np in enumerate(viz_data['camera_images'][:3]):
        y1, x1, y2, x2 = place_image(img_np, 0, i)
        camera_name = config.DATA_USED[i] if i < len(config.DATA_USED) else f'Camera_{i+1}'
        add_text(f'{camera_name}', x1 + 10, y1 + 30, color=(255, 255, 255))
        if i == 1:  # Central camera
            add_text(f'{viz_data["direction_command"]}', x1 + 10, y1 + 60)
            # Denormalize speed [-1, 1] -> [min, max] from config
            speed_normalized = viz_data['speed']
            speed_range = g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0]
            speed_denorm = speed_normalized * speed_range + g_conf.DATA_NORMALIZATION['speed'][0]
            # Add text
            add_text(f'Speed: {speed_denorm:.2f} m/s', x1 + 10, y1 + 90)
    
    # Attention Layer 1 (spans 3 columns)
    if len(viz_data['attention_maps']) > 0:
        overlay = create_overlay(viz_data['img_cat_np'], viz_data['attention_maps'][0])
        y1, x1, y2, x2 = place_image(overlay, 0, 3, col_span=3)
        add_text('Attention Layer 1', x1 + 10, y1 + 30, color=(255, 255, 255))
    
    # Row 2: ResNet Features (3 cols) + Attention Layer 2 (3 cols)
    for i, feature_map in enumerate(viz_data['resnet_maps'][:3][::-1]):
        y1, x1, y2, x2 = place_image(feature_map, 1, i, resize_mode=cv2.INTER_NEAREST, is_resnet=True)
        add_text('ResNet Block 4 - Reduction: "Max"', x1 + 10, y1 + 30, color=(255, 255, 255))
    
    if len(viz_data['attention_maps']) > 1:
        overlay = create_overlay(viz_data['img_cat_np'], viz_data['attention_maps'][1])
        y1, x1, y2, x2 = place_image(overlay, 1, 3, col_span=3)
        add_text('Attention Layer 2', x1 + 10, y1 + 30, color=(255, 255, 255))
    
    # Row 3: L2 Norm (3 cols) + Attention Layer 3 (3 cols)
    overlay = create_overlay(viz_data['img_cat_np'], viz_data['l2_resized'])
    y1, x1, y2, x2 = place_image(overlay, 2, 0, col_span=3)
    l2_min, l2_max = viz_data['l2_range']
    entropy = viz_data['l2_entropy']
    add_text(f'L2 Norm (Range: [{l2_min:.2f}, {l2_max:.2f}], Entropy: {entropy:.3f})', 
             x1 + 10, y1 + 30, color=(255, 255, 255))
    
    if len(viz_data['attention_maps']) > 2:
        overlay = create_overlay(viz_data['img_cat_np'], viz_data['attention_maps'][2])
        y1, x1, y2, x2 = place_image(overlay, 2, 3, col_span=3)
        add_text('Attention Layer 3', x1 + 10, y1 + 30, color=(255, 255, 255))
    
    # Row 4: PCA (3 cols) + Attention Layer 4 (3 cols)
    # PCA is special - it's already RGB, so handle differently
    pca_img = viz_data['pca_resized']
    if pca_img.dtype != np.uint8:
        pca_img = (pca_img * 255).astype(np.uint8)
    
    base_img = ensure_uint8_bgr(viz_data['img_cat_np'])
    pca_bgr = cv2.cvtColor(pca_img, cv2.COLOR_RGB2BGR)
    
    if base_img.shape != pca_bgr.shape:
        pca_bgr = cv2.resize(pca_bgr, (base_img.shape[1], base_img.shape[0]))
    
    overlay = cv2.addWeighted(base_img, 0.3, pca_bgr, 0.7, 0)
    
    y1, x1, y2, x2 = place_image(overlay, 3, 0, col_span=3)
    threshold = viz_data['threshold']
    add_text(f'PCA (Red: PC1, Green: PC2, Blue: PC3, Threshold: {threshold})', 
             x1 + 10, y1 + 30, color=(255, 255, 255))
    
    if len(viz_data['attention_maps']) > 3:
        overlay = create_overlay(viz_data['img_cat_np'], viz_data['attention_maps'][3])
        y1, x1, y2, x2 = place_image(overlay, 3, 3, col_span=3)
        add_text('Attention Layer 4', x1 + 10, y1 + 30, color=(255, 255, 255))
    
    # Row 5: Action Plots (steering: 2 cols, acceleration: 2 cols, trajectory: 2 cols)
    draw_action_plots_and_trajectory(canvas, 4, action_history, prediction_history, position_history, img_h, img_w, frame_num)
    
    return canvas


def draw_action_plots_and_trajectory(canvas, row, action_history, prediction_history, position_history, img_h, img_w, frame_num):
    """Draw action plots and 2D trajectory using OpenCV."""
    if len(action_history) < 2:
        return
    
    # Steering plot (left 2 cols)
    plot_x1, plot_y1 = 0, row * img_h
    plot_x2, plot_y2 = 2 * img_w, (row + 1) * img_h
    cv2.rectangle(canvas, (plot_x1, plot_y1), (plot_x2, plot_y2), (255, 255, 255), -1)
    
    margin = 50
    plot_area_x1, plot_area_y1 = plot_x1 + margin, plot_y1 + margin
    plot_area_x2, plot_area_y2 = plot_x2 - margin, plot_y2 - margin
    plot_area_w, plot_area_h = plot_area_x2 - plot_area_x1, plot_area_y2 - plot_area_y1
    
    draw_line_plot(canvas, action_history, prediction_history, 0,
                   plot_area_x1, plot_area_y1, plot_area_w, plot_area_h,
                   title="Steering History", frame_num=frame_num)
    
    # Acceleration plot (middle 2 cols)
    plot_x1, plot_y1 = 2 * img_w, row * img_h
    plot_x2, plot_y2 = 4 * img_w, (row + 1) * img_h
    cv2.rectangle(canvas, (plot_x1, plot_y1), (plot_x2, plot_y2), (255, 255, 255), -1)
    
    plot_area_x1, plot_area_y1 = plot_x1 + margin, plot_y1 + margin
    plot_area_x2, plot_area_y2 = plot_x2 - margin, plot_y2 - margin
    plot_area_w, plot_area_h = plot_area_x2 - plot_area_x1, plot_area_y2 - plot_area_y1
    
    draw_line_plot(canvas, action_history, prediction_history, 1,
                   plot_area_x1, plot_area_y1, plot_area_w, plot_area_h,
                   title="Acceleration History", frame_num=frame_num)
    
    # Trajectory plot (right 2 cols)
    plot_x1, plot_y1 = 4 * img_w, row * img_h
    plot_x2, plot_y2 = 6 * img_w, (row + 1) * img_h
    cv2.rectangle(canvas, (plot_x1, plot_y1), (plot_x2, plot_y2), (255, 255, 255), -1)
    
    plot_area_x1, plot_area_y1 = plot_x1 + margin, plot_y1 + margin
    plot_area_x2, plot_area_y2 = plot_x2 - margin, plot_y2 - margin
    plot_area_w, plot_area_h = plot_area_x2 - plot_area_x1, plot_area_y2 - plot_area_y1
    
    draw_trajectory_plot(canvas, position_history, 
                        plot_area_x1, plot_area_y1, plot_area_w, plot_area_h,
                        title="Vehicle Trajectory")

def draw_trajectory_plot(canvas, position_history, x, y, w, h, title, max_points: int = 1000, grid_spacing: float = 50.0):
    """Draw 2D trajectory plot with auto-centering and zoom."""
    if len(position_history) < 2:
        return
    
    # Intelligent sampling for very long trajectories
    if len(position_history) > max_points:
        # Keep recent points dense, older points sparse
        recent_dense = position_history[-max_points//2:]  # Last 500 points full density
        older_sparse = position_history[:-max_points//2:len(position_history)//max_points]  # Sample older points
        recent_positions = older_sparse + recent_dense
    else:
        recent_positions = position_history
    
    # Extract X and Y coordinates (assuming ego_position is [x, y, z])
    x_coords = [pos[0] for pos in recent_positions]
    y_coords = [pos[1] for pos in recent_positions]
    
    # Calculate bounds with padding
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding (20% on each side)
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    x_padding = x_range * 0.2
    y_padding = y_range * 0.2
    
    x_bounds = (x_min - x_padding, x_max + x_padding)
    y_bounds = (y_min - y_padding, y_max + y_padding)
    
    def calculate_spatial_grid_step(spatial_range, base_spacing=50.0):
        """Calculate appropriate grid spacing based on the spatial range."""
        # Similar logic to calculate_horizontal_grid_step but for spatial coordinates
        magnitude = 10 ** int(np.floor(np.log10(spatial_range)))
        normalized_range = spatial_range / magnitude
        
        if normalized_range <= 2:
            step = 0.5 * magnitude
        elif normalized_range <= 5:
            step = 1 * magnitude
        elif normalized_range <= 10:
            step = 2 * magnitude
        else:
            step = 5 * magnitude
    
        return max(step, base_spacing)  # Never go below base_spacing
    
    # Normalize coordinates to plot space
    def normalize_coords(pos_x, pos_y):
        norm_x = (pos_x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
        norm_y = (pos_y - y_bounds[0]) / (y_bounds[1] - y_bounds[0])
        
        plot_x = int(x + norm_x * w)
        plot_y = int(y + h - norm_y * h)  # Flip Y axis
        return plot_x, plot_y
    
    # Calculate dynamic grid spacing based on the view range
    dynamic_x_spacing = calculate_spatial_grid_step(x_bounds[1] - x_bounds[0])
    dynamic_y_spacing = calculate_spatial_grid_step(y_bounds[1] - y_bounds[0])
    
    # Draw fixed grid lines every dynamic_{x,y}_spacing meters
    # Vertical grid lines (X coordinates)
    x_grid_start = (int(x_bounds[0] / dynamic_x_spacing) - 1) * dynamic_x_spacing
    x_grid_pos = x_grid_start
    while x_grid_pos <= x_bounds[1]:
        if x_bounds[0] <= x_grid_pos <= x_bounds[1]:
            grid_x, _ = normalize_coords(x_grid_pos, y_bounds[0])
            cv2.line(canvas, (grid_x, y), (grid_x, y + h), (200, 200, 200), 1)
            
            # Add grid label
            cv2.putText(canvas, f'{int(x_grid_pos)}m', (grid_x - 15, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        x_grid_pos += dynamic_x_spacing
    
    # Horizontal grid lines (Y coordinates)
    y_grid_start = (int(y_bounds[0] / dynamic_x_spacing) - 1) * dynamic_x_spacing
    y_grid_pos = y_grid_start
    while y_grid_pos <= y_bounds[1]:
        if y_bounds[0] <= y_grid_pos <= y_bounds[1]:
            _, grid_y = normalize_coords(x_bounds[0], y_grid_pos)
            cv2.line(canvas, (x, grid_y), (x + w, grid_y), (200, 200, 200), 1)
            
            # Add grid label
            cv2.putText(canvas, f'{int(y_grid_pos)}m', (x - 40, grid_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        y_grid_pos += dynamic_x_spacing
    
    # Draw trajectory path
    for i in range(len(recent_positions) - 1):
        pt1 = normalize_coords(x_coords[i], y_coords[i])
        pt2 = normalize_coords(x_coords[i + 1], y_coords[i + 1])
        
        # Color gradient: older points are more transparent/blue, newer are red
        color_ratio = i / (len(recent_positions) - 1)
        color = (int(255 * (1 - color_ratio)), 0, int(255 * color_ratio))  # Blue to Red
        
        cv2.line(canvas, pt1, pt2, color, 2)
    
    # Draw current position (larger circle)
    if recent_positions:
        current_pos = normalize_coords(x_coords[-1], y_coords[-1])
        cv2.circle(canvas, current_pos, 5, (0, 255, 0), -1)  # Green circle
        
        # Add current position coordinates as text
        curr_x, curr_y = x_coords[-1], y_coords[-1]
        cv2.putText(canvas, f'({curr_x:.1f}, {curr_y:.1f})', 
                   (current_pos[0] + 10, current_pos[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 2)
    
    # Draw border
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 2)
    
    # Add title
    cv2.putText(canvas, title, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add scale info
    cv2.putText(canvas, f'Range: {x_range:.1f}m x {y_range:.1f}m', 
                (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)


def draw_line_plot(canvas, gt_data, pred_data, data_idx, x, y, w, h, title, frame_num):
    """Draw a line plot using OpenCV with dynamic grid and moving reference lines."""
    # Extract data
    gt_values = [action[data_idx] for action in gt_data]
    pred_values = [pred[data_idx] for pred in pred_data]
    
    # Calculate dynamic y-range from the data
    all_values = gt_values + pred_values
    y_min, y_max = min(all_values), max(all_values)
    
    # Add some padding (10% on each side)
    y_range_size = y_max - y_min
    if y_range_size == 0:
        y_range_size = 1  # Avoid division by zero
    y_padding = y_range_size * 0.1
    y_range = (y_min - y_padding, y_max + y_padding)
    
    # Normalize to plot coordinates
    def normalize_y(val):
        normalized = (val - y_range[0]) / (y_range[1] - y_range[0])
        return int(y + h - (normalized * h))  # Flip y-axis
    
    def normalize_x(idx):
        if len(gt_values) <= 1:
            return x + w // 2
        return int(x + (idx / (len(gt_values) - 1)) * w)
    
    # Calculate dynamic horizontal grid lines based on data range
    def calculate_horizontal_grid_step(data_range):
        """Calculate appropriate step size for horizontal grid lines."""
        range_size = data_range[1] - data_range[0]
        
        # Find appropriate step size (nice round numbers)
        magnitude = 10 ** int(np.floor(np.log10(range_size)))
        normalized_range = range_size / magnitude
        
        if normalized_range <= 1:
            step = 0.2 * magnitude
        elif normalized_range <= 2:
            step = 0.5 * magnitude
        elif normalized_range <= 5:
            step = 1 * magnitude
        else:
            step = 2 * magnitude
            
        return step
    
    # Draw dynamic horizontal grid lines
    h_step = calculate_horizontal_grid_step(y_range)
    grid_start = np.ceil(y_range[0] / h_step) * h_step
    
    grid_val = grid_start
    while grid_val <= y_range[1]:
        grid_y = normalize_y(grid_val)
        cv2.line(canvas, (x, grid_y), (x + w, grid_y), (200, 200, 200), 1)
        
        # Add value label
        cv2.putText(canvas, f'{grid_val:.3f}', (x - 45, grid_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        grid_val += h_step
    
    # Draw dynamic vertical grid lines (every 100 frames)
    tick_interval = 100
    current_frame = len(gt_values) - 1  # Current frame index
    
    # Calculate which tick marks should be visible
    start_tick = (current_frame // tick_interval) * tick_interval
    
    for tick in range(start_tick - tick_interval * 5, current_frame + tick_interval, tick_interval):
        if tick < 0:
            continue
            
        # Find the corresponding x position for this tick
        if tick <= current_frame:
            # Calculate position relative to the data we have
            relative_pos = tick - (current_frame - len(gt_values) + 1)
            if 0 <= relative_pos < len(gt_values):
                grid_x = normalize_x(relative_pos)
                cv2.line(canvas, (grid_x, y), (grid_x, y + h), (200, 200, 200), 1)
                
                # Add tick label
                cv2.putText(canvas, f'{tick}', (grid_x - 15, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    # Draw horizontal axis (y=0) if it's in range
    if y_range[0] <= 0 <= y_range[1]:
        zero_y = normalize_y(0)
        cv2.line(canvas, (x, zero_y), (x + w, zero_y), (80, 80, 80), 2)
    
    # Draw border
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 2)
    
    # Draw data lines
    if len(gt_values) > 1:
        # Ground truth (blue)
        for i in range(len(gt_values) - 1):
            pt1 = (normalize_x(i), normalize_y(gt_values[i]))
            pt2 = (normalize_x(i + 1), normalize_y(gt_values[i + 1]))
            cv2.line(canvas, pt1, pt2, (255, 0, 0), 3)  # Blue in BGR
        
        # Predictions (red)
        for i in range(len(pred_values) - 1):
            pt1 = (normalize_x(i), normalize_y(pred_values[i]))
            pt2 = (normalize_x(i + 1), normalize_y(pred_values[i + 1]))
            cv2.line(canvas, pt1, pt2, (0, 0, 255), 2)  # Red in BGR
    
    # Calculate the actual frame numbers
    history_length = len(gt_values)
    # current_actual_frame = frame_num  # Pass this as a parameter
    start_frame = frame_num - history_length + 1

    # Draw vertical grid lines every 50 frames
    for tick_frame in range((start_frame // 50) * 50, frame_num + 50, 50):
        if start_frame <= tick_frame <= frame_num:
            # Calculate position in the plot
            relative_pos = tick_frame - start_frame
            grid_x = normalize_x(relative_pos)
            cv2.line(canvas, (grid_x, y), (grid_x, y + h), (200, 200, 200), 1)
            cv2.putText(canvas, f'{tick_frame}', (grid_x - 15, y + h + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 2)
    
    # Add title (black text)
    cv2.putText(canvas, title, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add labels
    cv2.putText(canvas, "GT", (x + w - 100, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(canvas, "Pred", (x + w - 100, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

@main.command(name='visualize-model-inference')
@click.option('--dataset-path', type=click.Path(exists=True), required=True, help='Path to the dataset directory containing the route data')
@click.option('--route-path', type=str, required=True, help='Relative path to the specific route within dataset-path')
@click.option('--exp-batch', type=str, required=True, help='Experiment batch name for model configuration')
@click.option('--exp-name', type=str, required=True, help='Experiment name for model configuration')
@click.option('--checkpoint-path', type=click.Path(exists=True), required=True, help='Path to model checkpoints directory')
@click.option('--checkpoint-number', type=int, default=None, help='Specific checkpoint number to load (latest if not specified)')
@click.option('--video-name', type=str, required=True, help='Name for the output video (without extension)')
@click.option('--fps', type=float, default=10.0, help='Frames per second for the output video')
@click.option('--history-seconds', type=float, default=5.0, help='Seconds of action history to show in plots')
@click.option('--output-path', type=click.Path(), default=None, help='Output directory for videos (default: dataset_path/videos)')
@click.option('--frame-range', type=str, default=None, help='Frame range to process (e.g., "100-200" or "150")')
@click.option('--rgb-prefix', type=str, default='rgb', help='Prefix for RGB image files')
@click.option('--canbus-prefix', type=str, default='il_data_rosbag2', help='Prefix for CAN bus data files')
@click.option('--num-workers', type=int, default=4, help='Number of parallel workers for processing')
def visualize_model_inference(dataset_path, route_path, exp_batch, exp_name, 
                            checkpoint_path, checkpoint_number, video_name, fps, 
                            history_seconds, output_path, frame_range, rgb_prefix, 
                            canbus_prefix, num_workers):
    """
    Create visualization videos for model inference using OpenCV (much faster).
    """
    
    # Setup paths (existing code)
    full_route_path = os.path.join(dataset_path, route_path)
    print(f'Scanning data in {full_route_path}')
    if output_path is None:
        output_path = os.path.join(dataset_path, 'videos')
    os.makedirs(output_path, exist_ok=True)
    
    # Load model configuration and checkpoint (existing code)
    print(f"Loading model configuration: {exp_batch}/{exp_name}")
    eval_utils.load_config(exp_batch, exp_name)
    
    model = CIL_multiview(g_conf.MODEL_CONFIGURATION)
    checkpoint_path = os.path.join(checkpoint_path, '_results', exp_batch, exp_name, 'checkpoints')
    eval_utils.load_model_from_checkpoint(model, checkpoint_path, checkpoint_number, remove_pos_enc=False)
    model.eval()
    model = model.to('cuda')
    print("Model loaded successfully")
    
    # Get files and frame groups (existing code)
    image_extensions = ['.png', '.jpg', '.jpeg']
    # Filter out real () or synthetic attention prefixes
    # TODO: finish this!
    image_prefixes = [prefix for prefix in g_conf.DATA_USED if prefix != canbus_prefix]
    
    json_files = get_files_with_prefix_and_suffix(full_route_path, [canbus_prefix], ['.json'])
    image_files = get_files_with_prefix_and_suffix(full_route_path, g_conf.DATA_USED, image_extensions)
    all_files = json_files + image_files
    
    frame_groups = group_files_by_sorted_order(all_files, g_conf.DATA_USED)
    complete_frames = frame_groups
    
    if not complete_frames:
        print("No complete frames found with all required sensors")
        return
    
    # Apply frame range filter (existing code)
    if frame_range:
        if '-' in frame_range:
            start, end = map(int, frame_range.split('-'))
            complete_frames = {f: d for f, d in complete_frames.items() if start <= f <= end}
        else:
            frame_num = int(frame_range)
            complete_frames = {f: d for f, d in complete_frames.items() if f >= frame_num}
    
    sorted_frames = sorted(complete_frames.keys())
    print(f"Found {len(sorted_frames)} complete frames to process")
    
    if not sorted_frames:
        print("No frames to process")
        return
    
    # Calculate history length
    history_length = int(fps * history_seconds)
    
    # ========== NEW: Batch inference with precomputation ==========
    print("Batch processing model inference and precomputing visualizations...")
    
    frames_data = [complete_frames[frame_num] for frame_num in sorted_frames]
    inference_results = batch_model_inference_with_precompute(frames_data, model, batch_size=16)
    
    # ========== NEW: Fast OpenCV-based visualization creation ==========
    print("Creating visualizations with OpenCV...")
    processed_frames = []
    action_history = deque(maxlen=history_length)
    prediction_history = deque(maxlen=history_length)
    position_history = deque(maxlen=history_length)
    
    # Build all action histories first
    all_action_histories = []
    all_prediction_histories = []
    all_position_histories = []
    action_history = deque(maxlen=history_length)
    prediction_history = deque(maxlen=history_length)
    position_history = []
    
    for result in inference_results:
        action_history.append(result['gt_action'])
        prediction_history.append(result['pred_action'])
        position_history.append(result['ego_position'])
        all_action_histories.append(list(action_history))
        all_prediction_histories.append(list(prediction_history))
        all_position_histories.append(list(position_history))
    
    # Parallel OpenCV visualization (OpenCV is thread-safe for this)
    def create_single_opencv_viz(args):
        frame_num, viz_data, action_hist, pred_hist, position_history = args
        try:
            return frame_num, create_opencv_visualization_frame(
                frame_num, action_hist, pred_hist, position_history, g_conf, viz_data
            ), True
        except Exception as e:
            print(f"Error creating visualization for frame {frame_num}: {e}")
            return frame_num, None, False
    
    print(f'Sorted frames: {len(sorted_frames)}, Inference results: {len(inference_results)}')
    print(f'Action histories: {len(all_action_histories)}, Prediction histories: {len(all_prediction_histories)}')
    print(f'Position histories: {len(all_position_histories)}')
    viz_args = [
        (sorted_frames[i], inference_results[i]['viz_data'], 
         all_action_histories[i], all_prediction_histories[i], all_position_histories[i])
        for i in range(len(sorted_frames))
    ]
    
    processed_frames = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(create_single_opencv_viz, args) for args in viz_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Creating visualizations", dynamic_ncols=True):
            frame_num, vis_frame, success = future.result()
            if success and vis_frame is not None:
                processed_frames.append((frame_num, vis_frame))
    
    # Sort by frame number to maintain order
    processed_frames.sort(key=lambda x: x[0])
    
    
    if not processed_frames:
        print("No frames were processed successfully")
        return
    
    # ========== Video creation (unchanged) ==========
    print(f"Creating video from {len(processed_frames)} processed frames...")
    
    video_filename = f"{video_name}_model_inference_fps{fps}.mp4"
    video_path = os.path.join(output_path, video_filename)
    
    # Get frame dimensions from first processed frame
    first_frame = processed_frames[0][1]
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame_num, vis_frame in tqdm(processed_frames, desc="Writing video", dynamic_ncols=True):
        # vis_frame is already in BGR format from OpenCV
        video_writer.write(vis_frame)
    
    video_writer.release()
    
    print(f"Video saved to: {video_path}")
    print(f"Video contains {len(processed_frames)} frames at {fps} FPS")
    print(f"Duration: {len(processed_frames)/fps:.2f} seconds")


# ====================== Entry point ======================


if __name__ == '__main__':
    main()

# =========================================================
