import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os

# os.environ['FORCE_TF_AVAILABLE'] = '1'

from glob import glob
from typing import Union, List, Tuple, Type
from tqdm import tqdm
import click
from dataloaders import transforms
from PIL import Image
import math
import cv2
from multiprocessing import Pool
from _utils import utils
import json
import shutil
import re

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.models.mask2former import modeling_mask2former, image_processing_mask2former
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

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


def get_paths(data_root: str, sensors: list = ['can_bus', 'depth', 'ss']) -> list:
    # Let's get all the paths for ALL the files in the dataset
    paths = glob(os.path.join(data_root, '**', '*'), recursive=True)
    # Filter out with the sensors + only files
    paths = [path for path in paths if (any(sensor in path for sensor in sensors) and os.path.isfile(path))]
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

def process_map(args) -> None:
    idx, noise_cat, depth_paths, semantic_segmentation_paths, depth_threshold, min_depth, num_data_route, base_path, route, converter_label = args
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
        converter_label=converter_label
    )
    *_, mask_merge_right = transforms.get_virtual_attention_map(
        depth_path=depth_paths[idx + num_data_route * 2],
        segmented_path=semantic_segmentation_paths[idx + num_data_route * 2],
        noise_cat=noise_cat,
        depth_threshold=depth_threshold,
        converter_label=converter_label
    )

    # Set the name of the virtual attention files
    fname_central = f'virtual_attention_central_'
    fname_left = f'virtual_attention_left_'
    fname_right = f'virtual_attention_right_'
    
    # Add the noise, if the noise category is different from 0 (no noise)
    fname_central = f'f{fname_central}noise_{noise_cat}_' if noise_cat != 0 else fname_central
    fname_left = f'{fname_left}noise_{noise_cat}_' if noise_cat != 0 else fname_left
    fname_right = f'{fname_right}noise_{noise_cat}_' if noise_cat != 0 else fname_right

    # Add the label converter and the index/frame number
    fname_central = f'{fname_central}{idx:06d}.jpg' if converter_label is None else f'{fname_central}{converter_label}{idx:06d}.jpg'
    fname_left = f'{fname_left}{idx:06d}.jpg' if converter_label is None else f'{fname_left}{converter_label}{idx:06d}.jpg'
    fname_right = f'{fname_right}{idx:06d}.jpg' if converter_label is None else f'{fname_right}{converter_label}{idx:06d}.jpg'

    # Save the masks, they are 2D numpy arrays, so we can use PIL
    Image.fromarray(mask_merge_central).save(os.path.join(base_path, route, fname_central))
    Image.fromarray(mask_merge_left).save(os.path.join(base_path, route, fname_left))
    Image.fromarray(mask_merge_right).save(os.path.join(base_path, route, fname_right))


def process_container(args) -> type(None):
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

    # left_images = sorted([path for path in paths if 'left' in path])
    # central_images = sorted([path for path in paths if 'central' in path])
    # right_images = sorted([path for path in paths if 'right' in path])
    # can_bus = sorted([path for path in paths if json_filename in path])
    # num_data_route = len(can_bus)

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

    # Create the videos by horizontally concatenating the 3 cameras
    for frame_number in sorted_frames:
        left_img = cv2.imread(left_images[frame_number])
        central_img = cv2.imread(central_images[frame_number])
        right_img = cv2.imread(right_images[frame_number])

        # Get the speed, steering, acceleration, command from the can bus
        with open(can_bus[frame_number]) as json_: 
            data = json.load(json_)
            speed = data['speed']  # [0, 1] adim
            steering = data['steer']  # [-1, 1] adim
            acceleration = data['acceleration']  # [0, 1] adim
            command = command_sign_dict[data['direction']]  # string
        
        # # Get the frame number from the filename
        # pattern = r'(\d+)(?=\.\w+$)'
        # match = re.search(pattern, left_images[idx])
        # frame_number = int(match.group(1))

        # Write the frame idx in the left camera
        cv2.putText(left_img, f'Frame: {frame_number:06d}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255), 2)
        # Input: Add the command at the top of the central camera
        cv2.putText(central_img, f'{command}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 0, 0), 2)
        # Input: Add the speed at the top of the right camera
        cv2.putText(right_img, f'Speed: {speed:.2f} m/s', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 0, 0), 2)
        # Output: Add the steering and acceleration at the bottom of the left and right cameras
        cv2.putText(left_img, f'Steering: {steering:.2f}', (10, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 255), thickness=2)
        cv2.putText(right_img, f'Acceleration: {acceleration:.2f}', (10, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 255), 2)
        
        concat_img = cv2.hconcat([left_img, central_img, right_img])

        video.write(concat_img)
        
    video.release()
    print(f"Video for {weather}/{route} created successfully.")


def resize_image(args: tuple) -> None:
    """ Resize an image to the given target size at the original directory. """
    image_path, target_size = args
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, target_size)
    
    # Extracting the directory and filename
    directory = os.path.dirname(image_path)
    filename = 'resized_' + os.path.basename(image_path)
    
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

# Get all files in a directory with a specific prefix, recursively
# def get_files_with_prefix(directory, prefixes):
#     file_paths = []
#     for root, dirs, files in os.walk(directory):
#         for f in files:
#             if any(f.startswith(prefix) for prefix in prefixes):
#                 file_paths.append(os.path.join(root, f))
#     return file_paths

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

def save_segmentation(predicted_semantic_maps, image_paths, save_name_start, extension):
    for seg_map, image_path in zip(predicted_semantic_maps, image_paths):
        segmentation = seg_map.cpu().numpy()
        rgb_image = mapillary_to_cityscapes_rgb(segmentation)
        
        dir_name, base_name = os.path.split(image_path)
        name, ext = os.path.splitext(base_name)
        ext = extension if extension is not None else ext  # Extension override
        output_name = f"{save_name_start}_{name.split('_')[-1]}{ext}"
        output_path = os.path.join(dir_name, output_name)
        
        img = Image.fromarray(rgb_image)
        img.save(output_path)

def save_segmentation_threaded(predicted_semantic_maps, image_paths, save_name_start, extension, num_workers=8):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(save_segmentation, [seg_map], [image_path], save_name_start, extension)
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
@click.option('--rgb-prefix', default='rgb', help='Prefixes of the RGB images to predict the semantic segmentation (e.g., rgb_central000124.png)', type=str)
# Data saving options
@click.option('--save-name-prefix', 'save_name_prefix', default='ss_hat', help='Prefix for the saved semantic segmentation images.', type=str)
@click.option('--save-extension', 'save_name_extension', default=None, help='Image extension for the saved semantic segmentation images.', type=click.Choice(['.png', '.jpg', '.jpeg']))
# Optional
@click.option('--device', default='cuda', help='Device to use for prediction.', type=click.Choice(['cuda', 'cpu']))
@click.option('--num-workers', default=8, help='Number of workers to use for parallel processing.', type=click.IntRange(min=1))
@click.option('--gpu-id', default=None, help='GPU ID to use for prediction, if using gpu.', type=click.IntRange(min=0))
@click.option('--batch-size', default=16, help='Batch size for prediction.', type=click.IntRange(min=1))
def predict_semantic_segmentation(dataset_path, rgb_prefix, save_name_prefix, save_name_extension, device, num_workers, gpu_id, batch_size):
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else '0'
    device = 'cuda' if gpu_id is not None and torch.cuda.is_available() else device

    model_name = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    image_paths = get_files_with_prefix(dataset_path, [rgb_prefix])
    utils.sort_nicely(image_paths)
    total_images = len(image_paths)

    dataset = ImageDataset(image_paths, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=3)
    
    with tqdm(total=total_images, desc="Processing images", unit="images", dynamic_ncols=True) as pbar:
        for batch in dataloader:
            predicted_semantic_maps, image_paths = process_batch(batch, processor, model, device)
            save_segmentation_threaded(predicted_semantic_maps, image_paths, save_name_prefix, save_name_extension, num_workers)
            pbar.update(len(image_paths))

    print('Done!')


@main.command(name='visualize-routes')
@click.option('--dataset-path', default='carla', help='Dataset root to visualize.', type=click.Path(exists=True))
@click.option('--fps', default=10.0, help='FPS of the video.', type=click.FloatRange(min=1.0))
@click.option('--camera-name', default='rgb', help='String in the camera/sensor name to use for the video', type=click.Choice(['rgb', 'resized_rgb', 'virtual_attention', 'noise_1', 'noise_2', 'noise_3', 'ss', 'ss_hat']), show_default=True)
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
# Additional params
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU.', type=click.IntRange(min=1))
@click.option('--debug', is_flag=True, help='Debug mode.')
def prepare_ss(dataset_path, processes_per_cpu: int = 1, debug: bool = False) -> type(None):
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
                paths = get_paths(data_root=os.path.join(dataset_path, subdata, route))
                
                # Let's get the paths for the 3 cameras of depth and ss, as well as the can bus
                semantic_segmentation_paths = [path for path in paths if 'ss' in path]

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
    depth_paths = [path for path in paths if 'depth' in path] if not ignore_depth else InfiniteNone()
    semantic_segmentation_paths = [path for path in paths if 'ss' in path]
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
@click.option('--max-depth', 'depth_threshold', default=20.0, help='Filter out objects beyond this depth.', type=click.FloatRange(min=0.0), show_default=True)
@click.option('--min-depth', 'min_depth', default=2.3, help='Filter out objects starting from this depth for the central camera. Default takes into account the hood of the car, if shown in the central camera.', type=click.FloatRange(min=0.0), show_default=True)
# Virtual attention maps options
@click.option('--converter-label', 'converter_label', default=None, help='Label to convert the semantic segmentation to.', type=click.Choice(['traffic', 'dynamic', 'static']))
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

    sensor_names = ['can_bus', ss_name] if ignore_depth else ['can_bus', 'depth', ss_name]

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
@click.option('--route-path', help='Path to the root of your route to modify', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
@click.option('--remove-ticks', help='Ranges of ticks to remove or int; ranges will be removed (inclusive), int will be the max tick left in data (not included)', required=True)
def clean_route(route_path: Union[str, os.PathLike], remove_ticks: str):
    """ Remove all files containing ticks greater than the given maximum tick in their file name.
        This is useful for removing the data from a specific route where the ego vehicle crashes, for
        example. WARNING: this is permanent, double check before running this command!!! """
    # First, find all the files in the route
    tick_ranges = parse_tick_ranges(remove_ticks)
    all_files = find_files(route_path, tick_ranges)

    # Ask the user one last time if they are sure; give the number of files to be deleted
    # as well as the route path
    print(f'Are you sure you want to delete {len(all_files)} files from {route_path}?')
    print('Type "yes" to confirm, anything else to cancel.')
    user_input = input()
    if user_input != 'yes':
        print('Aborting...')
        return
    
    # Delete the files
    for file in all_files:
        os.remove(file)

    print('Done!')


@main.command(name='resize-dataset')
@click.option('--dataset-path', help='Path to the root of your dataset to modify', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
@click.option('--res', 'target_resolution', help='Resolution (widthxheight) to resize the images to.', type=click.STRING, required=True)
@click.option('--img-ext', 'ext', default='png', help='Image extension to look for.', type=click.STRING, show_default=True)
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU.', type=click.IntRange(min=1))
def resize_dataset(dataset_path: Union[str, os.PathLike], target_resolution: str, ext: str = 'png', processes_per_cpu: int = 1):
    """ Resize the dataset to a given resolution. """
    # Find all RGB images
    rgb_images = [img for img in glob(os.path.join(dataset_path, '**', '*', f'rgb*.{ext}'), recursive=True)]

    # Get the target size
    target_size = tuple(map(int, target_resolution.split('x')))

    args = [(img, target_size) for img in rgb_images]

    with Pool(os.cpu_count() * processes_per_cpu) as pool:
        for _ in tqdm(pool.imap(resize_image, args), total=len(rgb_images), desc=f'Resizing the images', dynamic_ncols=True):
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


# ====================== Entry point ======================


if __name__ == '__main__':
    main()

# =========================================================
