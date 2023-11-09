import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from glob import glob
from tqdm import tqdm
import click
from dataloaders import transforms
from PIL import Image
import cv2
from multiprocessing import Pool


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


@click.group()
def main():
    pass


@main.command(name='prepare-ss')
@click.option('--dataset', default='carla', help='Dataset root to convert.', type=click.Path(exists=True))
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
@click.option('--dataset', default='carla', help='Dataset root to convert.', type=click.Path(exists=True))
@click.option('--max-depth', 'depth_threshold',default=20.0, help='Filter out objects beyond this depth.', type=click.FloatRange(min=0.0))
@click.option('--min-depth', 'min_depth', default=2.3, help='Filter out objects starting from this depth for the central camera.', type=click.FloatRange(min=0.0))
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


if __name__ == '__main__':
    main()
