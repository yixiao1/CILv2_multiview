import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from glob import glob
from typing import Union, List, Tuple
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

# ====================== Helper functions ======================

def get_paths(data_root: str, sensors: list = ['can_bus', 'depth', 'ss']) -> list:
    # Let's get all the paths for ALL the files in the dataset
    paths = glob(os.path.join(data_root, '**', '*'), recursive=True)
    # Filter out with the sensors + only files
    paths = [path for path in paths if (any(sensor in path for sensor in sensors) and os.path.isfile(path))]
    # Sort the paths
    return sorted(paths)


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

def process_map(args) -> type(None):
    idx, depth_paths, semantic_segmentation_paths, depth_threshold, min_depth, num_data_route, dataset, subdata, route = args
    *_, mask_merge_central = transforms.get_virtual_attention_map(
        depth_path=depth_paths[idx],
        segmented_path=semantic_segmentation_paths[idx],
        depth_threshold=depth_threshold,
        min_depth=min_depth,
        central_camera=True
    )
    *_, mask_merge_left = transforms.get_virtual_attention_map(
        depth_path=depth_paths[idx + num_data_route],
        segmented_path=semantic_segmentation_paths[idx + num_data_route],
        depth_threshold=depth_threshold
    )
    *_, mask_merge_right = transforms.get_virtual_attention_map(
        depth_path=depth_paths[idx + num_data_route * 2],
        segmented_path=semantic_segmentation_paths[idx + num_data_route * 2],
        depth_threshold=depth_threshold
    )

    # Save the masks, they are 2D numpy arrays, so we can use PIL
    Image.fromarray(mask_merge_central).save(os.path.join(dataset, subdata, route, f'virtual_attention_central_{idx:06d}.jpg'))
    Image.fromarray(mask_merge_left).save(os.path.join(dataset, subdata, route, f'virtual_attention_left_{idx:06d}.jpg'))
    Image.fromarray(mask_merge_right).save(os.path.join(dataset, subdata, route, f'virtual_attention_right_{idx:06d}.jpg'))


def process_container(args) -> type(None):
    container_path, dataset_path = args
    container = container_path.split(os.sep)[-1]

    json_path_list = glob(os.path.join(container_path, 'can_bus*.json'))
    utils.sort_nicely(json_path_list)
    command_list=[]
    dist=[]
    for json_file in json_path_list:
        with open(json_file, 'r') as json_:
            data = json.load(json_)
            command = data['direction']
            command_list.append(command)
            dist.append(max(data['speed'], 0.0)* 0.1)

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

            with open(os.path.join(dataset_path, container, f'cmd_fix_{json_file.split(os.sep)[-1]}'), 'w') as fd:
                json.dump(pseudo_data, fd, indent=4, sort_keys=True)
        else:
            shutil.copy(json_file, os.path.join(dataset_path, container, 'cmd_fix_' + json_file.split('/')[-1]))



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
        check if tick is greater than it for elimination."""
    if isinstance(ranges, int):
        return tick > ranges
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
 

def create_video_for_route(dataset_path, weather, route, fps, camera_name):
    
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
                      sensors=[camera_name, 'cmd_fix_can_bus'])

    assert len(paths) % 4 == 0, f"Error, missing some data!"

    left_rgb = sorted([path for path in paths if 'left' in path])
    central_rgb = sorted([path for path in paths if 'central' in path])
    right_rgb = sorted([path for path in paths if 'right' in path])
    can_bus = sorted([path for path in paths if 'cmd_fix_can_bus' in path])
    num_data_route = len(can_bus)

    # We will use the central camera as the reference for the video size
    central_img = cv2.imread(central_rgb[0])
    height, width, _ = central_img.shape

    # Setup the video writer
    if not os.path.exists(os.path.join(dataset_path, 'videos')):
        os.makedirs(os.path.join(dataset_path, 'videos', weather), exist_ok=True)
    video_name = os.path.join(dataset_path, 'videos', weather, f'{route}.mp4')
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (3 * width, height))

    # Create the videos by horizontally concatenating the 3 cameras
    for idx in range(num_data_route):
        left_img = cv2.imread(left_rgb[idx])
        central_img = cv2.imread(central_rgb[idx])
        right_img = cv2.imread(right_rgb[idx])

        # Get the speed, steering, acceleration, command from the can bus
        with open(can_bus[idx]) as json_: 
            data = json.load(json_)
            speed = data['speed']  # [0, 1] adim
            steering = data['steer']  # [-1, 1] adim
            acceleration = data['acceleration']  # [0, 1] adim
            command = command_sign_dict[data['direction']]  # string

        # Write the frame idx in the left camera
        cv2.putText(left_img, f'Frame: {idx}', (10, 30), 
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


# ====================== Main functions ======================


@click.group()
def main():
    pass


# ============================================================


@main.command(name='visualize-routes')
@click.option('--dataset-path', default='carla', help='Dataset root to visualize.', type=click.Path(exists=True))
@click.option('--fps', default=10.0, help='FPS of the video.', type=click.FloatRange(min=1.0))
@click.option('--camera-name', default='rgb', help='Beginning of the camera name', type=click.Choice(['rgb', 'resized_rgb']), show_default=True)
# Additional params
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU.', type=click.IntRange(min=1))
def visualize_routes(dataset_path, fps, camera_name, processes_per_cpu: int = 1) -> type(None):
    """ Generate one video per route in the dataset. The structure of the dataset is as follows: 
            data_root/WEATHER/ROUTE/SENSOR_DATA 
        where WEATHER is one of four weather types (ClearNoon, etc.), ROUTE is the route number, and
        SENSOR_DATA is the sensor data for that route, ordered in time and starting in 00000.
        We will save the videos at the root of the dataset, in a folder called 'videos'.
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
         # Create the videos folder
        os.makedirs(os.path.join(dataset_path, 'videos', weather), exist_ok=True)

        routes = sorted([route for route in os.listdir(os.path.join(dataset_path, weather)) if os.path.isdir(os.path.join(dataset_path, weather, route))])
        for route in routes:
            pool.apply_async(create_video_for_route, args=(dataset_path, weather, route, fps, camera_name))

    pool.close()
    pool.join()

    print('Done!')


@main.command(name='prepare-ss')
@click.option('--dataset-path', default='carla', help='Dataset root to convert.', type=click.Path(exists=True))
# Additional params
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU.', type=click.IntRange(min=1))
@click.option('--debug', is_flag=True, help='Debug mode.')
def prepare_ss(dataset, processes_per_cpu: int = 1, debug: bool = False) -> type(None):
    """ Convert the dataset's semantic segmentation images to RGB if they are not (if only one channel has all the info) """
    # First, start with getting all the subdirectories in the dataset; usual structure: data_root/subroute/route_00001/rgb_central06d.jpg, for example
    subdatasets = sorted([subdir for subdir in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, subdir))])
    print('Subdatasets found: ', subdatasets) if debug else None

    with Pool(os.cpu_count() * processes_per_cpu) as pool:
        for subdata in subdatasets:
            # Get the routes in the subdataset
            routes = sorted([route for route in os.listdir(os.path.join(dataset, subdata)) if os.path.isdir(os.path.join(dataset, subdata, route))])
            print('Routes found: ', routes) if debug else None

            for route in routes:
                # Get the sensor data paths
                paths = get_paths(data_root=os.path.join(dataset, subdata, route))
                
                # Let's get the paths for the 3 cameras of depth and ss, as well as the can bus
                semantic_segmentation_paths = [path for path in paths if 'ss' in path]

                args = [(path, dataset, subdata, route) for path in semantic_segmentation_paths]
                for _ in tqdm(pool.imap(prepare_semantic_segmentation, args), total=len(args), desc=f'Preparing the semantic segmentation images [{subdata}/{route}]'):
                    pass

    print('Done!')


@main.command(name='create-virtual-attentions')
@click.option('--dataset-path', default='carla', help='Dataset root to convert.', type=click.Path(exists=True))
@click.option('--max-depth', 'depth_threshold', default=20.0, help='Filter out objects beyond this depth.', type=click.FloatRange(min=0.0), show_default=True)
@click.option('--min-depth', 'min_depth', default=2.3, help='Filter out objects starting from this depth for the central camera. Default takes into account the hood of the car, if shown in the central camera.', type=click.FloatRange(min=0.0), show_default=True)
# Additional params
@click.option('--processes-per-cpu', 'processes_per_cpu', default=1, help='Number of processes per CPU.', type=click.IntRange(min=1))
@click.option('--debug', is_flag=True, help='Debug mode.')
def create_virtual_atts(dataset, depth_threshold, min_depth, processes_per_cpu: int = 1, debug: bool = False) -> type(None):
    """ Generate the virtual attention maps for the dataset using the depth and semantic segmentation images. """
    # First, start with getting all the subdirectories in the dataset; usual structure: data_root/subroute/route_00001/rgb_central06d.jpg, for example
    subdatasets = sorted([subdir for subdir in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, subdir))])
    print('Subdatasets found: ', subdatasets) if debug else None

    with Pool(os.cpu_count() * processes_per_cpu) as pool:
        for subdata in subdatasets:
            # Get the routes in the subdataset
            routes = sorted([route for route in os.listdir(os.path.join(dataset, subdata)) if os.path.isdir(os.path.join(dataset, subdata, route))])
            print('Routes found: ', routes) if debug else None

            for route in routes:
                # Get the sensor data paths
                paths = get_paths(data_root=os.path.join(dataset, subdata, route))
                
                # Let's get the paths for the 3 cameras of depth and ss, as well as the can bus
                depth_paths = [path for path in paths if 'depth' in path]
                semantic_segmentation_paths = [path for path in paths if 'ss' in path]
                can_bus_paths = [path for path in paths if path.split(os.sep)[-1].startswith('can_bus')]

                assert len(depth_paths) == len(semantic_segmentation_paths) == len(can_bus_paths) * 3, \
                    f"Error, sensor mismatch: number of Depth paths: {len(depth_paths)}, SS paths: {len(semantic_segmentation_paths)}, CAN Bus paths: {len(can_bus_paths)}"

                num_data_route = len(can_bus_paths)

                # Prepare the semantic segmentation images before

                args = [(idx, depth_paths, semantic_segmentation_paths, depth_threshold, min_depth, num_data_route, dataset, subdata, route) for idx in range(num_data_route)]
                for _ in tqdm(pool.imap(process_map, args), total=num_data_route, desc=f'Generating the virtual attention maps [{subdata}/{route}]'):
                    pass

    print('Done!')


@main.command(name='command-fix')
@click.option('--dataset-path', help='Path to the root of your dataset to modify', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
def command_fix(dataset_path: Union[str, os.PathLike]):
    """ Manually fix a bug in the dataset wherein the command/direction is given too soon to the ego vehicle. """
    all_containers_path_list = glob(os.path.join(dataset_path, '*'))
    utils.sort_nicely(all_containers_path_list)

    args = [(container_path, dataset_path) for container_path in all_containers_path_list]

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap(process_container, args), total=len(all_containers_path_list)):
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
        for _ in tqdm(pool.imap(resize_image, args), total=len(rgb_images), desc=f'Resizing the images'):
            pass

# ====================== Entry point ======================


if __name__ == '__main__':
    main()

# =========================================================
