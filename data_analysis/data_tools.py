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

def create_model_visualization_frame(model, data_dict: dict, frame_num: int, 
                                   action_history: deque, prediction_history: deque,
                                   config, resnet_inter, attn_weights, encoder_output) -> np.ndarray:
    """
    Create a comprehensive visualization frame for model inference.
    Layout: 6 columns x 5 rows as specified.
    """
    # Create the visualization grid
    fig = plt.figure(figsize=(24, 20))
    
    # Configure matplotlib for better video output
    plt.rcParams.update({'font.size': 10})
    
    # Get source images for display
    src_images = [data_dict[camera_type] for camera_type in config.DATA_USED 
                  if any(cam_type in camera_type for cam_type in ['rgb', 'sekonix', 'conti'])]
    
    # ========== Row 1: RGB Images (3 cols) + Attention Layer 1 (3 cols) ==========
    for i, img in enumerate(src_images[:3]):  # Ensure max 3 cameras
        ax = plt.subplot2grid((5, 6), (0, i), colspan=1)
        img_denorm = TF.normalize(img, 
                             [-0.485/0.229, -0.456/0.224, -0.406/0.255], 
                             [1/0.229, 1/0.224, 1/0.255])
        img_np = torch.clamp(img_denorm, 0, 1).detach().cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img_np)
        camera_name = config.DATA_USED[i] if i < len(config.DATA_USED) else f'Camera_{i+1}'
        ax.set_title(f'{camera_name}\n(Frame {frame_num:06d})')
        ax.axis('off')
    
    # Attention Layer 1 (spans 3 columns)
    if len(attn_weights) > 0:
        ax = plt.subplot2grid((5, 6), (0, 3), colspan=3)
        img_cat = torch.cat(src_images[:3], dim=2)
        img_cat_norm = TF.normalize(img_cat, [-0.485/0.229, -0.456/0.224, -0.406/0.255], 
                              [1/0.229, 1/0.224, 1/0.255])
        img_cat_np = torch.clamp(img_cat_norm, 0, 1).cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img_cat_np)
        
        attn = attn_weights[0]
        attn_avg = attn.squeeze().mean(dim=0)
        num_cameras = len(src_images[:3])
        
        # Remove register tokens and action tokens if present
        start_idx = 0
        if model.num_register_tokens > 0:
            start_idx += model.num_register_tokens
        if not config.NO_ACT_TOKENS:
            start_idx += 2
        if config.CMD_SPD_TOKENS:
            start_idx += 2
            
        # Get only spatial tokens
        spatial_tokens = attn_avg[start_idx:]
        attn_reshaped = rearrange(spatial_tokens, 
                                 '(h w cam) -> 1 h (w cam)' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else '(cam h w) -> 1 h (cam w)', 
                                 cam=num_cameras, h=model.res_out_h)
        
        attn_np = attn_reshaped.detach().cpu().numpy().transpose(1, 2, 0)
        attn_resized = cv2.resize(attn_np, (img_cat_np.shape[1], img_cat_np.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
        
        ax.imshow(attn_resized, cmap='jet', alpha=0.6)
        ax.set_title('Attention Layer 1')
        ax.axis('off')
    
    # ========== Row 2: ResNet Features (3 cols) + Attention Layer 2 (3 cols) ==========
    resnet_features = resnet_inter[-1]
    reduction_type = 'max'
    
    for i in range(min(3, len(src_images))):
        ax = plt.subplot2grid((5, 6), (1, i), colspan=1)
        features = resnet_features[i]
        feature_map = torch.max(features, dim=0)[0] if reduction_type == 'max' else torch.mean(features, dim=0)
        
        im = ax.imshow(feature_map.detach().cpu().numpy(), cmap='jet')
        ax.set_title(f'ResNet {reduction_type.title()}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Attention Layer 2
    if len(attn_weights) > 1:
        ax = plt.subplot2grid((5, 6), (1, 3), colspan=3)
        ax.imshow(img_cat_np)
        
        attn = attn_weights[1]
        attn_avg = attn.squeeze().mean(dim=0)
        # Get only spatial tokens
        spatial_tokens = attn_avg[start_idx:]
        attn_reshaped = rearrange(spatial_tokens, 
                                 '(h w cam) -> 1 h (w cam)' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else '(cam h w) -> 1 h (cam w)', 
                                 cam=num_cameras, h=model.res_out_h)
        
        attn_np = attn_reshaped.detach().cpu().numpy().transpose(1, 2, 0)
        attn_resized = cv2.resize(attn_np, (img_cat_np.shape[1], img_cat_np.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
        
        ax.imshow(attn_resized, cmap='jet', alpha=0.6)
        ax.set_title('Attention Layer 2')
        ax.axis('off')
    
    # ========== Row 3: L2 Norm (3 cols) + Attention Layer 3 (3 cols) ==========
    ax = plt.subplot2grid((5, 6), (2, 0), colspan=3)
    ax.imshow(img_cat_np)
    
    # Calculate L2 norm and entropy
    spatial_features = encoder_output.squeeze()[start_idx:]  # Remove tokens
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
    l2_resized = cv2.resize(l2_np_reshaped, (img_cat_np.shape[1], img_cat_np.shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
    
    ax.imshow(l2_resized, cmap='jet', alpha=0.6)
    ax.set_title(f'L2 Norm (Range: [{l2_norm.min():.2f}, {l2_norm.max():.2f}], Entropy: {entropy:.3f})')
    ax.axis('off')
    
    # Attention Layer 3
    if len(attn_weights) > 2:
        ax = plt.subplot2grid((5, 6), (2, 3), colspan=3)
        ax.imshow(img_cat_np)
        
        attn = attn_weights[2]
        attn_avg = attn.squeeze().mean(dim=0)
        # Get only spatial tokens
        spatial_tokens = attn_avg[start_idx:]
        attn_reshaped = rearrange(spatial_tokens, 
                                 '(h w cam) -> 1 h (w cam)' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else '(cam h w) -> 1 h (cam w)', 
                                 cam=num_cameras, h=model.res_out_h)
        
        attn_np = attn_reshaped.detach().cpu().numpy().transpose(1, 2, 0)
        attn_resized = cv2.resize(attn_np, (img_cat_np.shape[1], img_cat_np.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
        
        ax.imshow(attn_resized, cmap='jet', alpha=0.6)
        ax.set_title('Attention Layer 3')
        ax.axis('off')
    
    # ========== Row 4: PCA (3 cols) + Attention Layer 4 (3 cols) ==========
    ax = plt.subplot2grid((5, 6), (3, 0), colspan=3)
    ax.imshow(img_cat_np)
    
    # Compute PCA with threshold
    threshold = 0  # Can be made configurable
    spatial_size, latent_dim = spatial_features.shape
    features = spatial_features.reshape(-1, latent_dim).detach().cpu().numpy()
    spatial_size = features.shape[0]
    
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    
    bg_mask = pca_features[:, 0] < threshold
    fg_mask = ~bg_mask
    
    # PCA for only foreground patches
    pca_features_fg = pca.transform(features[fg_mask])
    for i in range(3):
        pca_features_fg[:, i] = minmax_scale(pca_features_fg[:, i])
        
    pca_features_rgb = pca_features.copy()
    pca_features_rgb[bg_mask] = 0
    pca_features_rgb[fg_mask] = pca_features_fg
    
    # Reshape for visualization
    pca_reshaped = rearrange(pca_features_rgb, 
                            '(h w cam) c -> c h (w cam) ' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else '(cam h w) c -> c h (cam w)',
                            cam=num_cameras, h=model.res_out_h, c=3)
    pca_np = pca_reshaped.transpose(1, 2, 0)
    pca_resized = cv2.resize(pca_np, (img_cat_np.shape[1], img_cat_np.shape[0]), 
                            interpolation=cv2.INTER_LINEAR)
    
    # Clip to valid range to avoid matplotlib warnings
    pca_resized = np.clip(pca_resized, 0, 1)
    
    ax.imshow(pca_resized, alpha=0.7)
    ax.set_title(f'PCA (Red: PC1, Green: PC2, Blue: PC3, Threshold: {threshold})')
    ax.axis('off')
    
    # Attention Layer 4 (or last layer)
    if len(attn_weights) > 3:
        ax = plt.subplot2grid((5, 6), (3, 3), colspan=3)
        ax.imshow(img_cat_np)
        
        attn = attn_weights[3] if len(attn_weights) > 3 else attn_weights[-1]
        attn_avg = attn.squeeze().mean(dim=0)
        # Get only spatial tokens
        spatial_tokens = attn_avg[start_idx:]
        attn_reshaped = rearrange(spatial_tokens, 
                                 '(h w cam) -> 1 h (w cam)' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else '(cam h w) -> 1 h (cam w)', 
                                 cam=num_cameras, h=model.res_out_h)
        
        attn_np = attn_reshaped.detach().cpu().numpy().transpose(1, 2, 0)
        attn_resized = cv2.resize(attn_np, (img_cat_np.shape[1], img_cat_np.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
        
        ax.imshow(attn_resized, cmap='jet', alpha=0.6)
        layer_name = 'Attention Layer 4'
        ax.set_title(layer_name)
        ax.axis('off')
    
    # ========== Row 5: Action Plots ==========
    # Steering angle plot (3 cols)
    ax = plt.subplot2grid((5, 6), (4, 0), colspan=3)
    if len(action_history) > 1:
        frames = list(range(len(action_history)))
        gt_steers = [action[0] for action in action_history]
        pred_steers = [pred[0] for pred in prediction_history]
        
        ax.plot(frames, gt_steers, 'b-', label='Ground Truth', linewidth=2)
        ax.plot(frames, pred_steers, 'r--', label='Prediction', linewidth=2)
        ax.axvline(x=len(frames)-1, color='k', linestyle=':', alpha=0.7, label='Current')
        ax.legend()
        
        # Fix axis limits to prevent movement
        ax.set_xlim(0, max(len(frames), 10))  # Ensure minimum width
        ax.set_ylim(-1.1, 1.1)  # Fixed steering range
        
    ax.set_title('Steering Angle History')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Steering')
    ax.grid(True, alpha=0.3)
    
    # Acceleration plot (3 cols)
    ax = plt.subplot2grid((5, 6), (4, 3), colspan=3)
    if len(action_history) > 1:
        gt_accels = [action[1] for action in action_history]
        pred_accels = [pred[1] for pred in prediction_history]
        
        ax.plot(frames, gt_accels, 'b-', label='Ground Truth', linewidth=2)
        ax.plot(frames, pred_accels, 'r--', label='Prediction', linewidth=2)
        ax.axvline(x=len(frames)-1, color='k', linestyle=':', alpha=0.7, label='Current')
        ax.legend()

        # Fix axis limits
        ax.set_xlim(0, max(len(frames), 10))
        ax.set_ylim(-1.1, 1.1)  # Fixed acceleration range    

    ax.set_title('Acceleration History')
    ax.set_xlabel('Frames')
    ax.set_ylabel('Acceleration')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return buf

def process_single_inference_frame(args):
    """Process a single frame for model inference visualization."""
    (frame_num, frame_data, model, config, action_history, prediction_history, 
     history_length, output_dir) = args
    
    try:
        # Load data directly from the file paths
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

        # Load images for each sensor in DATA_USED
        for sensor_type in g_conf.DATA_USED:
            if sensor_type in frame_data:
                img = eval_utils.open_image(os.path.dirname(frame_data[sensor_type]), 
                                        os.path.basename(frame_data[sensor_type]))
                datapoint[sensor_type] = img

        data = train_transform(datapoint, tuple(g_conf.IMAGE_SHAPE))
        
        # Get model prediction with all intermediate outputs
        with torch.no_grad():
            action_output, resnet_inter, attn_weights, encoder_output = eval_utils.model_forward(
                model, data, last_encoder_state=True
            )
        pred_action = action_output.squeeze().detach().cpu().numpy().tolist()

        # Update history
        action_history.append(gt_action)
        prediction_history.append(pred_action)

        # Keep only recent history
        if len(action_history) > history_length:
            action_history.popleft()
            prediction_history.popleft()

        # Create visualization with pre-computed model outputs
        vis_frame = create_model_visualization_frame(
            model, data, frame_num, action_history, prediction_history, config,
            resnet_inter, attn_weights, encoder_output
        )
        
        return frame_num, vis_frame, True
        
    except Exception as e:
        print(f"Error processing frame {frame_num}: {e}")
        return frame_num, None, False

@main.command(name='visualize-model-inference')
@click.option('--dataset-path', type=click.Path(exists=True), required=True,
              help='Path to the dataset directory containing the route data')
@click.option('--route-path', type=str, required=True,
              help='Relative path to the specific route within dataset-path')
@click.option('--exp-batch', type=str, required=True,
              help='Experiment batch name for model configuration')
@click.option('--exp-name', type=str, required=True,
              help='Experiment name for model configuration')
@click.option('--checkpoint-path', type=click.Path(exists=True), required=True,
              help='Path to model checkpoints directory')
@click.option('--checkpoint-number', type=int, default=None,
              help='Specific checkpoint number to load (latest if not specified)')
@click.option('--video-name', type=str, required=True,
              help='Name for the output video (without extension)')
@click.option('--fps', type=float, default=10.0, 
              help='Frames per second for the output video')
@click.option('--history-seconds', type=float, default=5.0,
              help='Seconds of action history to show in plots')
@click.option('--output-path', type=click.Path(), default=None,
              help='Output directory for videos (default: dataset_path/videos)')
@click.option('--frame-range', type=str, default=None,
              help='Frame range to process (e.g., "100-200" or "150")')
@click.option('--rgb-prefix', type=str, default='rgb',
              help='Prefix for RGB image files')
@click.option('--canbus-prefix', type=str, default='cmd_fix_can_bus',
              help='Prefix for CAN bus data files')
@click.option('--num-workers', type=int, default=4,
              help='Number of parallel workers for processing')
def visualize_model_inference(dataset_path, route_path, exp_batch, exp_name, 
                            checkpoint_path, checkpoint_number, video_name, fps, 
                            history_seconds, output_path, frame_range, rgb_prefix, 
                            canbus_prefix, num_workers):
    """
    Create visualization videos for model inference showing ResNet features, 
    attention maps, L2 norms, PCA, and action predictions vs ground truth.
    """
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Setup paths
    full_route_path = os.path.join(dataset_path, route_path)
    print(f'Scanning data in {full_route_path}')
    if output_path is None:
        output_path = os.path.join(dataset_path, 'videos')
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Loading model configuration: {exp_batch}/{exp_name}")
    
    # Load model configuration and checkpoint
    eval_utils.load_config(exp_batch, exp_name)
    
    model = CIL_multiview(g_conf.MODEL_CONFIGURATION)
    checkpoint_path = os.path.join(checkpoint_path, '_results', exp_batch, exp_name, 'checkpoints')
    eval_utils.load_model_from_checkpoint(model, checkpoint_path, checkpoint_number, remove_pos_enc=False)
    model.eval()
    model = model.to('cuda')
    
    print("Model loaded successfully")
    
    # Filter frames that have all required sensors
    image_extensions = ['.png', '.jpg', '.jpeg']
    json_files = get_files_with_prefix_and_suffix(full_route_path, ['il_data'], ['.json'])
    image_files = get_files_with_prefix_and_suffix(full_route_path, g_conf.DATA_USED, image_extensions)
    all_files = json_files + image_files
    
    frame_groups = group_files_by_sorted_order(all_files, g_conf.DATA_USED)
    complete_frames = frame_groups  # Already filtered in the grouping function
    
    if not complete_frames:
        print("No complete frames found with all required sensors")
        return
    
    # Apply frame range filter if specified
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
    
    # Calculate history length in frames
    history_length = int(fps * history_seconds)
    
    # Initialize history tracking
    action_history = deque(maxlen=history_length)
    prediction_history = deque(maxlen=history_length)
    
    # Prepare arguments for parallel processing
    temp_dir = os.path.join(output_path, f'temp_{video_name}')
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Processing frames...")
    
    # Process frames in order (sequential for history tracking)
    processed_frames = []
    
    for frame_num in tqdm(sorted_frames, desc="Processing frames"):
        args = (frame_num, complete_frames[frame_num], model, g_conf, 
               action_history, prediction_history, history_length, temp_dir)

        frame_num, vis_frame, success = process_single_inference_frame(args)
                
        if success and vis_frame is not None:
            processed_frames.append((frame_num, vis_frame))
    
    if not processed_frames:
        print("No frames were processed successfully")
        return
    
    print(f"Creating video from {len(processed_frames)} processed frames...")
    
    # Create video from processed frames
    video_filename = f"{video_name}_model_inference_fps{fps}.mp4"
    video_path = os.path.join(output_path, video_filename)
    
    # Get frame dimensions from first processed frame
    first_frame = processed_frames[0][1]
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame_num, vis_frame in tqdm(processed_frames, desc="Writing video"):
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
    
    video_writer.release()
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Video saved to: {video_path}")
    print(f"Video contains {len(processed_frames)} frames at {fps} FPS")
    print(f"Duration: {len(processed_frames)/fps:.2f} seconds")


# ====================== Entry point ======================


if __name__ == '__main__':
    main()

# =========================================================
