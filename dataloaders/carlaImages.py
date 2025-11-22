import os
import json
import math
import numpy as np
from PIL import Image
from torch.utils import data
from dataloaders.transforms import train_transform, val_transform, canbus_normalization

from configs import g_conf
from typing import Union, List, Dict
import re


class carlaImages(data.Dataset):

    def __init__(self, model_name, base_dir, dataset_list, split="train", rank: int = 0,
                 resize_attention: 'tuple[int]' = (10, 10)):
        self.root = base_dir
        self.split = split
        self.data = []
        self.data_in_chunk = []
        self.model_name = model_name
        self.dataset_name = ''
        self.resize_attention = resize_attention

        for dataset_name in dataset_list:
            self.dataset_name += dataset_name.split(os.sep)[-1]

        for dataset_name in dataset_list:
            self.images_base = os.path.join(self.root, dataset_name)
            #### For different models, we set different strategy for loading data, and we save the npy file for next time better loading
            canbus_paths = self.recursive_glob(rootdir=self.images_base, prefix='cmd_fix', suffix='.json')
            all_cam_paths_dict = {}
            for camera_type in g_conf.DATA_USED:
                img_paths = self.get_data_paths_with_fallback(
                    rootdir=self.images_base,
                    prefix=camera_type,
                    suffixes=['.png', '.jpg'] if any(gaze_type in camera_type for gaze_type in ['gaze_pred', 'scout']) else ['.jpg', '.png'],
                    avoid='noise' if ('virtual_attention' in camera_type and g_conf.ATTENTION_NOISE_CATEGORY == 0) else None
                )
                if 'virtual_attention' in camera_type:
                    # Filter out data, as double safety for the avoid parameter above
                    ext = os.path.splitext(img_paths[0])[1] if img_paths else '.jpg'
                    img_paths = [path for path in img_paths if re.match(f'{camera_type}\d{{6}}{re.escape(ext)}', os.path.basename(path))]
                all_cam_paths_dict.update({camera_type: img_paths})

            self.data = self._add_canbus_data_point(self.data, all_cam_paths_dict, canbus_paths)

            # with multiple frames input we also need to ensure the frames are from the same episode
            self.data_in_chunk = self.get_episode_chunk(self.data_in_chunk, rootdir=self.images_base,
                                                        prefix='cmd_fix', suffix='.json')

        index_list = list(range(0, len(self.data)))
        index_chunks = []
        count = 0
        for chunk in self.data_in_chunk:
            index_chunks.append(index_list[count:count + len(chunk)])
            count += len(chunk)

        self.block_index_start = []
        self.block_index_end = []
        block_num_start = (g_conf.ENCODER_INPUT_FRAMES_NUM - 1) * g_conf.ENCODER_STEP_INTERVAL
        block_num_end = (g_conf.DECODER_OUTPUT_FRAMES_NUM - g_conf.ENCODER_INPUT_FRAMES_NUM + g_conf.ENCODER_OUTPUT_STEP_DELAY) * g_conf.ENCODER_STEP_INTERVAL

        if block_num_start > 0:
            for chunk in index_chunks:
                self.block_index_start += chunk[:block_num_start]
        if block_num_end > 0:
            for chunk in index_chunks:
                self.block_index_end += chunk[-block_num_end:]

        if rank == 0:
            print(split)
            print(f'  - number of chunks: {len(index_chunks)}')
            print(f'  - block number of data per chunk: begining: {block_num_start} end: {block_num_end}')
            print(f'  - total data blocked: {len(self.block_index_start + self.block_index_end)}')

    def __len__(self):
        return len(self.data)

    def get_data_paths_with_fallback(self, rootdir, prefix, suffixes, avoid=None):
        """
        Try to find files with given suffixes in order, returning the first non-empty result.
        
        Args:
            rootdir: Root directory to search
            prefix: File prefix to match
            suffixes: List of file suffixes to try in order (e.g., ['.jpg', '.png', '.npy'])
            avoid: Optional string to avoid in filenames
            
        Returns:
            List of file paths
            
        Raises:
            RuntimeError: If no files found with any of the provided suffixes
        """
        for suffix in suffixes:
            paths = self.recursive_glob(rootdir=rootdir, prefix=prefix, suffix=suffix, avoid=avoid)
            if len(paths) > 0:
                return paths
        
        # If no files found with any suffix, raise an error
        raise RuntimeError(
            f"No files found for prefix '{prefix}' with any of the suffixes {suffixes} in {rootdir}"
        )

    def analyze_index(self, index):
        if index in self.block_index_start:
            index += (g_conf.ENCODER_INPUT_FRAMES_NUM - 1) * g_conf.ENCODER_STEP_INTERVAL

        elif index in self.block_index_end:
            index -= (g_conf.DECODER_OUTPUT_FRAMES_NUM - g_conf.ENCODER_INPUT_FRAMES_NUM + g_conf.ENCODER_OUTPUT_STEP_DELAY) * g_conf.ENCODER_STEP_INTERVAL

        return index

    def __getitem__(self, index):
        # We try to avoid "list out of range problem". Since we may use more than one frame for inputs or outputs, the index should not be in the last few points
        # Besides, we also make these frames to come from the same episode, which means that they need to be sequential
        index = self.analyze_index(index)

        data_vec = {'current': [], 'future': []}
        for n in range(g_conf.ENCODER_INPUT_FRAMES_NUM):
            datapoint = self.data[index - (g_conf.ENCODER_INPUT_FRAMES_NUM - 1 - n) * g_conf.ENCODER_STEP_INTERVAL]
            sample = {'can_bus': datapoint['can_bus']}
            for camera_type in g_conf.DATA_USED:
                if 'virtual_attention' in camera_type:
                    img = Image.open(datapoint[camera_type]).convert('L')
                elif any(gaze_type in camera_type for gaze_type in ['gaze_pred', 'scout']):
                    img = Image.open(datapoint[camera_type]).convert('L')
                # TODO: sensor type, not always rgb
                else:
                    img = Image.open(datapoint[camera_type]).convert('RGB')
                sample.update({camera_type: img})

            if self.split == 'train':
                one_frame_data = self.transform_tr(sample, resize_attention=self.resize_attention)
                data_vec['current'].append(one_frame_data)

            elif self.split == 'val':
                one_frame_data = self.transform_val(sample, resize_attention=self.resize_attention)
                data_vec['current'].append(one_frame_data)

        # the output has time delay
        if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
            for o in range(g_conf.DECODER_OUTPUT_FRAMES_NUM):
                datapoint_future = self.data[index - (
                            g_conf.ENCODER_INPUT_FRAMES_NUM - 1 - o - g_conf.ENCODER_OUTPUT_STEP_DELAY) * g_conf.ENCODER_STEP_INTERVAL]
                sample_future = {'can_bus_future': datapoint_future['can_bus']}
                data_vec['future'].append(sample_future)
                ### We comment the future image to speed up training
                # img_future = Image.open(datapoint_future['image']).convert('RGB')
                # sample_future = {'image_future': img_future, 'can_bus_future': datapoint_future['can_bus']}
                # if self.split == 'train':
                #     one_frame_data_future = self.transform_tr(sample_future)
                #     data_vec['future'].append(one_frame_data_future)

                # elif self.split == 'val':
                #     one_frame_data_future = self.transform_val(sample_future)
                #     data_vec['future'].append(one_frame_data_future)

                # del one_frame_data_future
                # del img_future
            del sample_future
            del datapoint_future

        del one_frame_data
        del sample
        del datapoint
        del img

        return data_vec

    def get_valid_ticks_for_episode(self, episode_path, camera_types, canbus_prefix='cmd_fix'):
        """
        Find all ticks where ALL required sensors and canbus data exist.
        
        Args:
            episode_path: Path to the episode folder (e.g., root/weather/Route00001)
            camera_types: List of required camera/sensor types from g_conf.DATA_USED
            canbus_prefix: Prefix for canbus files
            
        Returns:
            List of valid tick numbers (sorted)
        """
        # Extract tick numbers for each sensor type
        tick_sets = {}
        
        # Get canbus ticks
        canbus_files = [f for f in os.listdir(episode_path) 
                        if f.startswith(canbus_prefix) and f.endswith('.json')]
        canbus_ticks = set()
        for f in canbus_files:
            # Extract tick from filename like "cmd_fix_canbus_000016.json"
            match = re.search(r'(\d{6})', f)  # TODO: this assumes specific naming of the files, must be general
            if match:
                canbus_ticks.add(int(match.group(1)))
        tick_sets['canbus'] = canbus_ticks
        
        # Get ticks for each camera/sensor type
        for camera_type in camera_types:
            # Determine file extension
            if any(gaze_type in camera_type for gaze_type in ['gaze_pred', 'scout']):
                suffix = '.png'
            elif 'virtual_attention' in camera_type:
                suffix = '.jpg'
            else:
                suffix = '.jpg'
            
            sensor_files = [f for f in os.listdir(episode_path) 
                        if f.startswith(camera_type) and f.endswith(suffix)]
            sensor_ticks = set()
            for f in sensor_files:
                # Extract tick from filename like "scout14ep1_000016.png"
                match = re.search(r'(\d{6})', f)
                if match:
                    sensor_ticks.add(int(match.group(1)))
            tick_sets[camera_type] = sensor_ticks
        
        # Find intersection of all tick sets (only ticks where ALL sensors exist)
        valid_ticks = set.intersection(*tick_sets.values()) if tick_sets else set()
        
        return sorted(list(valid_ticks))


    def _add_canbus_data_point(self, full_dataset, img_paths_dict, canbus_paths):
        """
        Add data points to the dataset, ensuring all sensors exist for each tick.
        """
        # Group paths by episode
        episodes = {}
        for canbus_path in canbus_paths:
            episode_dir = os.path.dirname(canbus_path)
            if episode_dir not in episodes:
                episodes[episode_dir] = {
                    'canbus': [],
                    'sensors': {cam_type: [] for cam_type in img_paths_dict.keys()}
                }
            episodes[episode_dir]['canbus'].append(canbus_path)
        
        # Group sensor paths by episode
        for camera_type, img_paths in img_paths_dict.items():
            for img_path in img_paths:
                episode_dir = os.path.dirname(img_path)
                if episode_dir in episodes:
                    episodes[episode_dir]['sensors'][camera_type].append(img_path)
        
        # Process each episode
        for episode_dir, episode_data in episodes.items():
            # Get valid ticks for this episode
            valid_ticks = self.get_valid_ticks_for_episode(
                episode_dir, 
                list(img_paths_dict.keys())
            )
            
            # Create a mapping of tick -> paths
            tick_to_paths = {}
            
            # Map canbus files
            for canbus_path in episode_data['canbus']:
                match = re.search(r'(\d{6})', os.path.basename(canbus_path))
                if match:
                    tick = int(match.group(1))
                    if tick in valid_ticks:
                        if tick not in tick_to_paths:
                            tick_to_paths[tick] = {'canbus': None, 'sensors': {}}
                        tick_to_paths[tick]['canbus'] = canbus_path
            
            # Map sensor files
            for camera_type, sensor_paths in episode_data['sensors'].items():
                for sensor_path in sensor_paths:
                    match = re.search(r'(\d{6})', os.path.basename(sensor_path))
                    if match:
                        tick = int(match.group(1))
                        if tick in valid_ticks:
                            tick_to_paths[tick]['sensors'][camera_type] = sensor_path
            
            # Create datapoints only for valid ticks
            for tick in sorted(tick_to_paths.keys()):
                paths = tick_to_paths[tick]
                
                # Verify all required data exists
                if paths['canbus'] is None:
                    continue
                if not all(cam_type in paths['sensors'] for cam_type in img_paths_dict.keys()):
                    continue
                
                # Create datapoint
                datapoint = {'can_bus': {}}
                
                # Load canbus data
                with open(paths['canbus'], 'r') as f:
                    canbus_data = json.loads(f.read())
                for value in g_conf.TARGETS + g_conf.OTHER_INPUTS:
                    datapoint['can_bus'][value] = canbus_data[value]
                datapoint['can_bus'] = canbus_normalization(
                    datapoint['can_bus'], 
                    g_conf.DATA_NORMALIZATION
                )
                
                # Add sensor paths
                for camera_type in img_paths_dict.keys():
                    datapoint[camera_type] = paths['sensors'][camera_type]
                
                full_dataset.append(datapoint)
        
        return full_dataset

    def recursive_glob(self, rootdir: Union[str, os.PathLike] = os.getcwd(), 
                       prefix: str = None, suffix: str = None, avoid: str = None,
                       exact_match: bool = False):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param prefix is the start prefix to be searched
            :param suffix is the suffix to be searched
        """
        if prefix is None:
            prefix = ''
        if suffix is None:
            suffix = ''
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in sorted(os.walk(rootdir))
                for filename in sorted(filenames) if filename.startswith(prefix) and filename.endswith(suffix) and (avoid is None or avoid not in filename)]

    def get_episode_chunk(self, data, rootdir='.', prefix='', suffix=''):
        for looproot, _, filenames in sorted(os.walk(rootdir)):
            files = []
            for filename in sorted(filenames):
                if filename.startswith(prefix) and filename.endswith(suffix):
                    files.append(os.path.join(looproot, filename))
            if files:
                data.append(files)
        return data

    def transform_tr(self, sample, resize_attention: 'tuple[int]'):
        return train_transform(sample, g_conf.IMAGE_SHAPE, resize_attention)

    def transform_val(self, sample, resize_attention: 'tuple[int]'):
        return val_transform(sample, g_conf.IMAGE_SHAPE, resize_attention)

class FlexibleCarlaDataset(data.Dataset):
    def __init__(self, model_name: str, base_dir: str, dataset_names: List[str], 
                 dataset_structure: str = 'carla_cil', split: str = "train", 
                 rank: int = 0):
        self.root = Path(base_dir)
        self.split = split
        self.data = []
        self.model_name = model_name
        
        # Get dataset structure
        self.structure = DATASET_STRUCTURES.get(dataset_structure)
        if not self.structure:
            raise ValueError(f"Unknown dataset structure: {dataset_structure}")
        
        # Load data from all datasets
        for dataset_name in dataset_names:
            dataset_path = self.root / dataset_name
            sequences = self.structure.get_data_paths(dataset_path)
            
            for seq in sequences:
                # Create data points from each sequence
                self._process_sequence(seq)
        
        # Handle temporal sequences
        self._create_chunks()
        
        if rank == 0:
            print(f"Loaded {len(self.data)} samples from {len(dataset_names)} datasets")
    
    def _process_sequence(self, sequence_data: Dict):
        """Process a single sequence and add to dataset"""
        num_frames = len(sequence_data['metadata'])
        
        for i in range(num_frames):
            datapoint = {
                'sequence_path': sequence_data['path'],
                'frame_idx': i,
                'metadata_file': sequence_data['metadata'][i],
                'sensors': {}
            }
            
            # Add sensor files
            for sensor_name, sensor_files in sequence_data['sensors'].items():
                if i < len(sensor_files):
                    datapoint['sensors'][sensor_name] = sensor_files[i]
            
            self.data.append(datapoint)
    
    def __getitem__(self, index):
        # Handle temporal indexing
        index = self.analyze_index(index)
        
        # Load current frame data
        datapoint = self.data[index]
        
        # Load metadata
        with open(datapoint['metadata_file']) as f:
            metadata = json.load(f)
        
        # Prepare sample
        sample = {
            'can_bus': self._process_canbus(metadata),
            'frame_idx': datapoint['frame_idx']
        }
        
        # Load sensor data
        for sensor_name, sensor_path in datapoint['sensors'].items():
            if sensor_path.exists():
                if 'rgb' in sensor_name:
                    img = Image.open(sensor_path).convert('RGB')
                elif 'depth' in sensor_name:
                    img = Image.open(sensor_path).convert('L')
                else:
                    img = Image.open(sensor_path)
                
                sample[sensor_name] = img
        
        # Apply transforms
        if self.split == 'train':
            return self.transform_tr(sample)
        else:
            return self.transform_val(sample)