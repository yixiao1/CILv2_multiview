#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from enum import Enum

import matplotlib.colors
from einops import rearrange

import carla
import os
import re
import io
import torch
import json
from typing import List, Tuple

import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from _utils import utils

from driving.utils.route_manipulation import downsample_route
from driving.envs.sensor_interface import SensorInterface

from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
from network.models.building_blocks.u2net import U2NET
from _utils.training_utils import DataParallelWrapper
from dataloaders.transforms import encode_directions_4, encode_directions_6, inverse_normalize, decode_directions_4, \
    decode_directions_6, get_virtual_noise_from_depth
from driving.utils.waypointer import Waypointer
from driving.utils.route_manipulation import interpolate_trajectory

from omegaconf import OmegaConf
from network.models.architectures.Roach_rl_birdview.birdview.chauffeurnet import ObsManager
from network.models.architectures.Roach_rl_birdview.utils.traffic_light import TrafficLightHandler
from importlib import import_module


def checkpoint_parse_configuration_file(filename):
    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    # If model doesn't have lens_circle_set, set it to False
    if 'lens_circle_set' not in configuration_dict:
        configuration_dict['lens_circle_set'] = False

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name'], configuration_dict['lens_circle_set']

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

def get_entry_point():
    return 'TED_agent'


class Track(Enum):
    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'


class TED_agent(object):
    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, save_driving_vision, save_driving_measurement, save_to_hdf5, plug_in_expert=False):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.waypointer = None
        self.attn_weights = None
        self.vision_save_path = save_driving_vision
        self.plug_in_expert=plug_in_expert

        # Data
        self.direction = None
        self.steer = None
        self.throttle = None
        self.brake = None

        self._model = None
        self.u2net = None
        self.norm_rgb = None
        self.norm_speed = None
        self.checkpoint = None
        self.world = None
        self.map = None

        # agent's initialization
        self.setup_model(path_to_conf_file)

        self.cmap_2 = plt.get_cmap('jet')
        self.cmap_1 = plt.get_cmap('Reds')
        self.datapoint_count = 0
        self.save_frequence = 1

    def setup_model(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """

        exp_dir = os.path.join('/', os.path.join(*path_to_conf_file.split('/')[:-1]))
        yaml_conf, checkpoint_number, _, self.lens_circle_setting = checkpoint_parse_configuration_file(path_to_conf_file)
        self.lens_circle_setting = (self.lens_circle_setting == "True")
        g_conf.immutable(False)
        merge_with_yaml(os.path.join(exp_dir, yaml_conf), process_type='drive')
        set_type_of_process('drive', root=os.environ["TRAINING_RESULTS_ROOT"])
        
        self._model = Models(g_conf.MODEL_CONFIGURATION, rank=0)
        if torch.cuda.device_count() > 1 and g_conf.DATA_PARALLEL:
            print("Using multiple GPUs parallel! ")
            print(torch.cuda.device_count(), 'GPUs to be used: ', os.environ["CUDA_VISIBLE_DEVICES"])
            self._model = DataParallelWrapper(self._model)
        self.checkpoint = torch.load(os.path.join(exp_dir, 'checkpoints', f'{self._model.name}_{checkpoint_number}.pth'))
        print(f'{self._model.name}_{checkpoint_number}.pth loaded from {os.path.join(exp_dir, "checkpoints")}')
        # Correctly load checkpoint if saved with module. Temporary fix, should be correctly saved during training!
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.checkpoint['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.checkpoint['model'] = new_state_dict


        if isinstance(self._model, torch.nn.DataParallel) or isinstance(self._model, torch.nn.parallel.DistributedDataParallel):
            self._model.module.load_state_dict(self.checkpoint['model'])
        else:
            self._model.load_state_dict(self.checkpoint['model'])
        self._model.cuda()
        self._model.eval()
        
        if g_conf.ATTENTION_AS_INPUT:
            self.u2net = U2NET(3, 1)
            u2net_model_dir = os.path.join(os.getcwd(), 'saved_models', 'u2net', 'u2net_bce_town01_8hrdata.pth')

            if torch.cuda.is_available():
                self.u2net.load_state_dict(torch.load(u2net_model_dir))
                self.u2net.cuda()
            else:
                self.u2net.load_state_dict(torch.load(u2net_model_dir, map_location='cpu'))
            self.u2net.eval()

        if self.plug_in_expert:
            self.setup_expert_agent(path_to_conf_file)

    def setup_expert_agent(self, path_to_conf_file):
        exp_dir = os.path.join('/', os.path.join(*path_to_conf_file.split('/')[:-3]), 'Roach_rl_birdview')
        cfg = OmegaConf.load(os.path.join(exp_dir, 'Roach_rl_birdview.yaml'))
        self._obs_managers = ObsManager(cfg['obs_configs']['birdview'])

    def set_world(self, world):
        self.world = world
        self.map = self.world.get_map()
        if self.plug_in_expert:
            TrafficLightHandler.reset(self.world)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """

        if self.plug_in_expert:
            self._route_plan = global_plan_world_coord

        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self.waypointer = Waypointer(self.world, global_plan_gps=self._global_plan, global_route=global_plan_world_coord)

    def reset_global_plan(self):
        """
        reset the plan (route) for the agent
        """
        current_loc = self._ego_vehicle.get_location()
        last_gps, _ = self._global_plan[-1]
        last_loc = self.waypointer.gps_to_location([last_gps['lat'], last_gps['lon'], last_gps['z']])
        gps_route, route = interpolate_trajectory(self.world, [current_loc, last_loc])

        if self.plug_in_expert:
            self._route_plan = route
            self._obs_managers.attach_ego_vehicle(self._ego_vehicle, self._route_plan)

        ds_ids = downsample_route(route, 50)
        self.route = [(route[x][0], route[x][1]) for x in ds_ids]
        self._global_plan = [gps_route[x] for x in ds_ids]
        self.waypointer.reset_route(global_plan_gps=self._global_plan, global_route=route)
        return route

    def set_ego_vehicle(self, ego_vehicle):
        self._ego_vehicle = ego_vehicle
        if self.plug_in_expert:
            self._obs_managers.attach_ego_vehicle(self._ego_vehicle, self._route_plan)

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        """

        bound_x = 2.9508416652679443  # For Lincoln MKZ 2017
        bound_y = 1.5641621351242065
        bound_z = 1.255373239517212

        if self.vision_save_path:
            sensors = [
                {'type': 'sensor.camera.rgb', 'x': -4.5, 'y': 0.0, 'z': 4.0, 'roll': 0.0, 'pitch': -20.0, 'yaw': 0.0,
                 'width': 1088, 'height': 680, 'fov': 120, 'id': 'rgb_backontop', 'lens_circle_setting': False},

                # RGB cameras
                {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 960, 'height': 540, 'fov': 45, 'id': 'rgb_central', 'lens_circle_setting': False},

                {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': -46,
                'width': 960, 'height': 540, 'fov': 45, 'id': 'rgb_left', 'lens_circle_setting': False},

                {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 46.0,
                'width': 960, 'height': 540, 'fov': 45, 'id': 'rgb_right', 'lens_circle_setting': False},

                {'type': 'sensor.other.gnss', 'id': 'GPS'},

                {'type': 'sensor.other.imu', 'id': 'IMU'},

                {'type': 'sensor.speedometer', 'id': 'SPEED'},

                {'type': 'sensor.can_bus', 'id': 'can_bus'}
            ]

        else:
            sensors = [
                # RGB cameras
                {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 960, 'height': 540, 'fov': 45, 'id': 'rgb_central', 'lens_circle_setting': False},

                {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': -46,
                'width': 960, 'height': 540, 'fov': 45, 'id': 'rgb_left', 'lens_circle_setting': False},

                {'type': 'sensor.camera.rgb', 'x': 0.0*bound_x+0.75, 'y': -0.2*bound_y, 'z': 1.0*bound_z-0.05, 'roll': 0.0, 'pitch': 0.0, 'yaw': 46.0,
                'width': 960, 'height': 540, 'fov': 45, 'id': 'rgb_right', 'lens_circle_setting': False},

                {'type': 'sensor.other.gnss', 'id': 'GPS'},

                {'type': 'sensor.other.imu', 'id': 'IMU'},

                {'type': 'sensor.speedometer', 'id': 'SPEED'},

                {'type': 'sensor.can_bus', 'id': 'can_bus'}
            ]

        return sensors

    def __call__(self, timestamp):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        self.input_data = self.sensor_interface.get_data()

        # for key, values in self.input_data.items():
        #     if values[0] != timestamp.frame:
        #         raise RuntimeError(' The frame number of sensor data does not match the timestamp frame:', key)

        if self.plug_in_expert:
            self.input_data = self.adding_BEV_data(self.input_data)

        control = self.run_step()
        control.manual_gear_shift = False

        return control

    def run_step(self):
        """
        Execute one step of navigation.
        :return: control
        """
        
        self.control = carla.VehicleControl()
        rgb_cameras = [c for c in g_conf.DATA_USED if 'rgb' in c]
        # Remove any other prefix (e.g., resized_rgb -> rgb, etc.)
        rgb_cameras = [re.sub(r'^.*?(rgb)', r'\1', c) for c in rgb_cameras]

        # Preprocess the RGB images
        self.norm_rgb = [[self.process_image(self.input_data[c][1]).unsqueeze(0).cuda() for c in rgb_cameras]]

        # Use the pre-trained UNet to get the attention maps as input
        if g_conf.ATTENTION_AS_INPUT:
            # For now, we assume the attention will come as a prediction from the pre-trained UNet
            # TODO: if g_conf.ATTENTION_FROM_UNET ...
            if g_conf.ATTENTION_NOISE_CATEGORY > 0:
                # If > 0, the attention map is noisy, so we need a depth camera
                depth_cameras = ['depth_left', 'depth_central', 'depth_right']

                self.depth_cameras = [self.process_depth(self.input_data[c][1], txt=c).unsqueeze(0).cuda() for c in depth_cameras]

            assert self.u2net is not None, 'No U2Net model loaded!'

            # Set the transform for each image
            tr = transforms.Compose([
                utils.RescaleT(320),
                utils.ToTensorLab()])
            

            # Get all the masks/synthetic attention maps
            for idx in range(len(rgb_cameras)):
                # This can be optimized, but whatever, they're just 3 cameras for now
                tr_img = tr(self.input_data[rgb_cameras[idx]][1]).unsqueeze(0)
                tr_img = tr_img.type(torch.FloatTensor)
                tr_img = torch.autograd.Variable(tr_img.cuda() if torch.cuda.is_available() else tr_img)

                # Predict the mask
                pred, *_ = self.u2net(tr_img)
                pred = pred.squeeze()
                mask = utils.min_max_norm(pred)  # type tensor, [320, 320], range [0, 1]
                mask = mask.unsqueeze(0).unsqueeze(0)  # type tensor, [1, 1, 320, 320], range [0, 1]

                # Add noise to the attention map
                if g_conf.ATTENTION_NOISE_CATEGORY > 0:
                    mask = mask * self.depth_cameras[idx]

                # Resize the mask to the desired shape
                if g_conf.ATTENTION_AS_NEW_CHANNEL:
                    # Append the attention map as a new channel to the input rgb image
                    mask = TF.resize(TF.normalize(mask, [0.5], [0.5]), [g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2]])
                    self.norm_rgb[0][idx] = torch.cat((self.norm_rgb[0][idx], mask), dim=1)
                else:
                    # Do an element-wise multiplication between the attention map and the input rgb image
                    mask = TF.resize(mask, [g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2]])
                    self.norm_rgb[0][idx] = self.norm_rgb[0][idx] * mask
        
        self.norm_speed = [torch.cuda.FloatTensor([self.process_speed(self.input_data['SPEED'][1]['speed'])]).unsqueeze(0)]
        #
        if g_conf.DATA_COMMAND_ONE_HOT:
            self.direction = \
                [torch.cuda.FloatTensor(self.process_command(self.input_data['GPS'][1],
                                                             self.input_data['IMU'][1])[0]).unsqueeze(0).cuda()]
        else:
            self.direction = \
                [torch.cuda.LongTensor([self.process_command(self.input_data['GPS'][1],
                                                             self.input_data['IMU'][1])[1]-1]).unsqueeze(0).cuda()]

        outputs = self._model.forward_eval(self.norm_rgb, self.direction, self.norm_speed,
                                           attn_rollout=False, attn_refinement=False)

        # Output of the model will be the actions, resnet features and the attention weights
        actions_outputs = outputs[0]
        self.attn_weights = outputs[-1]

        # Hand-crafted control outputs
        self.steer, self.throttle, self.brake = self.process_control_outputs(actions_outputs.detach().cpu().numpy().squeeze())

        # Pass to the controller
        self.control.steer = float(self.steer)
        self.control.throttle = float(self.throttle)
        self.control.brake = float(self.brake)
        self.control.hand_brake = False

        self.record_driving(self.input_data)
        self.datapoint_count += 1

        return self.control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        self._model = None
        self.norm_rgb = None
        self.norm_speed = None
        self.direction = None
        self.checkpoint = None
        self.world = None
        self.map = None
        self.attn_weights = None

        self.reset()

    def reset(self):
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = None
        self.input_data = None
        self.waypointer = None
        self.vision_save_path = None
        self.datapoint_count = 0

    def adding_BEV_data(self, input_dict):
        obs_dict = self._obs_managers.get_observation()
        input_dict.update({'birdview': obs_dict})
        return input_dict

    def process_depth(self, image, txt):
        image = image[:, :, ::-1]  # BGR to RGB
        image = Image.fromarray(image, mode='RGB')
        image = get_virtual_noise_from_depth(image, g_conf.ATTENTION_NOISE_CATEGORY, txt)
        image = TF.to_tensor(image)
        return image

    def process_image(self, image):
        image = Image.fromarray(image)
        image = image.resize((g_conf.IMAGE_SHAPE[2], g_conf.IMAGE_SHAPE[1])).convert('RGB')
        image = TF.to_tensor(image)
        # Normalization is really necessary if you want to use any pretrained weights.
        image = TF.normalize(image, mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'])
        return image

    def process_speed(self, speed):
        norm_speed = abs(speed - g_conf.DATA_NORMALIZATION['speed'][0]) / (
                g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0])  # [0.0, 1.0]
        return norm_speed

    def process_control_outputs(self, action_outputs):
        """ Hand-crafted control outputs processing. """
        if g_conf.ACCELERATION_AS_ACTION:
            steer, self.acceleration = action_outputs[0], action_outputs[1]
            if self.acceleration >= 0.0:
                throttle = self.acceleration
                brake = 0.0
            else:
                brake = np.abs(self.acceleration)
                throttle = 0.0
        else:
            steer, throttle, brake = action_outputs[0], action_outputs[1], action_outputs[2]
            if brake < 0.05:
                brake = 0.0

        return np.clip(steer, -1, 1), np.clip(throttle, 0, 1), np.clip(brake, 0, 1)

    def process_command(self, gps, imu):
        if g_conf.DATA_COMMAND_CLASS_NUM == 4:
            _, _, cmd = self.waypointer.tick_nc(gps, imu)
            return encode_directions_4(cmd.value), cmd.value
        elif g_conf.DATA_COMMAND_CLASS_NUM == 6:
            _, _, cmd = self.waypointer.tick_lb(gps, imu)
            return encode_directions_6(cmd.value), cmd.value

    def record_driving(self, current_input_data):
        if self.vision_save_path and self.datapoint_count % self.save_frequence == 0:
            # Aux function for getting the images to the desired shape
            def get_grayscale_attn_map(attn_weights: torch.Tensor,
                                       token_idx: int = None,
                                       resize_width: int = 300,
                                       resize_height: int = 300,
                                       one_seq: bool = False) -> np.ndarray:
                if token_idx is not None:
                    attn_weights = attn_weights[:, token_idx, :, :]  # [S, H, cam * W]
                attn_weights = attn_weights.detach().cpu().numpy()  # [S, H, cam * W] or [S, t+2, t+2]
                attn_weights = attn_weights.transpose(1, 2, 0)  # [H, cam * W, S] or [t+2, t+2, S]

                # Resize the attention map to the desired shape
                interp = cv2.INTER_CUBIC if one_seq else cv2.INTER_AREA
                attn_weights = cv2.resize(attn_weights, (resize_width, resize_height), interpolation=interp)  # [imgH, cam * imgW] or [imgH, imgW]

                # Take it back to the original shape (given S=1)
                attn_weights = attn_weights[:, :, None]  # [imgH, cam * imgW, 1] or [imgH, imgW, 1]
                attn_weights = attn_weights.transpose(2, 0, 1)  # [1, imgH, cam * imgW] or [1, imgH, imgW]

                return attn_weights

            def blend_gradcam_cameraimg(grayscale_cam: np.ndarray,
                                        cmap: matplotlib.colors.LinearSegmentedColormap,
                                        cams: List[Image.Image],
                                        cam_index: int,
                                        blend_strength: float = 0.5) -> List[Image.Image]:
                # TODO: this must work for any number of cams, not just one sequence
                cmap_att = np.delete(cmap(grayscale_cam), 3, 3)[0]
                cmap_att = Image.fromarray((cmap_att * 255).astype(np.uint8))
                blended_img = Image.blend(cams[cam_index], cmap_att, blend_strength)
                return [blended_img]

            def combine_attention_maps_to_rgb(data_maps, color_list):
                """Combines normalized 2D data maps into an RGB image based on a list of colors."""
                def hex_to_rgb(hex_color):
                    hex_color = hex_color.strip('#').lower()
                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

                def ensure_rgb(color):
                    if isinstance(color, str):  # Assuming the input is a hex color string
                        # Convert from hex to RGB tuple
                        return hex_to_rgb(color)
                    elif isinstance(color, tuple) and len(color) == 3:
                        # Already in RGB tuple format
                        return color
                    else:
                        raise ValueError("Invalid color format. Must be hexadecimal string or RGB tuple.")

                height, width = data_maps[0].shape
                # rgb_image = np.zeros((height, width, 3), dtype=np.float32)
                rgb_image = np.full((height, width, 3), 
                                    (53/255, 53/255, 49/255), dtype=np.float32)  # Black olive background

                # Ensure all colors are in RGB tuple format
                rgb_colors = [ensure_rgb(color) for color in color_list]

                # Add each color contribution
                for data_map, color in zip(data_maps, rgb_colors):
                    for c in range(3):  # There are three channels in RGB
                        rgb_image[:, :, c] += data_map * (color[c] / 255.0)  # Normalize RGB to [0,1]

                # Normalize to the range of 0-255
                rgb_image = np.clip(rgb_image * 255, 0, 255)  # Scale back up to RGB range
                return rgb_image.astype(np.uint8)
                

            def is_uniform_attention(head_weights: torch.Tensor, threshold: float = 1e-6) -> bool:
                """
                Check if the attention weights are approximately uniform.
                """
                seq_length = head_weights.numel()
                uniform_value = 1.0 / (seq_length - 10.0)
                return torch.allclose(head_weights, torch.full_like(head_weights, uniform_value), atol=threshold)

            def is_near_uniform_attention(head_weights: torch.Tensor, threshold: float = 0.05) -> bool:
                """
                Check if the attention weights are nearly uniform.
                We consider the distribution near-uniform if the difference between
                the maximum and minimum values is below the threshold.
                """
                min_val = head_weights.min()
                max_val = head_weights.max()
                return (max_val - min_val) < threshold

            def multihead_attention_viz(attn_weights: torch.Tensor,
                                        cams: List[Image.Image],
                                        blend_strength: float = 0.5) -> Tuple[Image.Image, List[Tuple[str, str]]]:
                
                # Define the ordered list of tuples for attention head names and indices
                ordered_head_names = [
                    ('Dynamic', 0), ('Traffic', 1), ('Human', -1), ('Static', 3),  # Coarse classes and human attention
                    ('Vehicle', 0), ('Pedestrian', 1), ('Traffic Light', 2), ('Pole', 3), ('Lane', 4),  # Fine-grained classes
                    ('Vehicle-L', 0), ('Pedestrian-L', 1), ('Trafficlight-L', 2), ('Pole-L', 3)  # Classes that include the lanes
                ]

                # Extract camera suffixes and get relevant attention heads
                camera_suffixes = utils.extract_camera_suffixes(g_conf.DATA_USED)
                relevant_heads = [utils.get_attention_head(suffix, g_conf) for suffix in camera_suffixes]

                # Get the total number of heads
                B, H, N1, N2 = attn_weights.shape

                # Replace -1 with H-1 for the last head
                relevant_heads = [H-1 if head == -1 else head for head in relevant_heads]

                # Color of each head (ordered as specified)
                idx_to_color = {
                    0: '#DC143C',  # crimson
                    1: '#007EA7',  # cerulean
                    2: '#FFC857',  # sunglow
                    3: '#6EBEA0',  # mint
                    4: '#F6BDD1',  # orchid pink
                    -1: '#4B0082', # indigo
                }

                # Select relevant attention maps
                relevant_attn_weights = attn_weights[0, relevant_heads, :, :]  # Use only the first batch

                # Reshape and normalize attention maps
                S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
                cam = len([c for c in g_conf.DATA_USED if 'rgb' in c])
                attn_maps = []
                head_info = []
                for i, (head_weights, suffix) in enumerate(zip(relevant_attn_weights, camera_suffixes)):
                    head_weights = head_weights.mean(dim=0)  # Average over N1 dimension
                    
                    # Check if attention is uniform and skip if so
                    if is_near_uniform_attention(head_weights):
                        continue
                    
                    head_weights = utils.min_max_norm(head_weights)
                    
                    # Use einops.rearrange for reshaping
                    if g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_LOSS:
                        head_weights = rearrange(head_weights, '(h w S cam) -> h (w S cam)',
                                                h=self._model._model.res_out_h, w=self._model._model.res_out_w, S=S, cam=cam)
                    else:
                        head_weights = rearrange(head_weights, '(S cam h w) -> h (S cam w)',
                                                S=S, cam=cam, h=self._model._model.res_out_h, w=self._model._model.res_out_w)
                    
                    attn_maps.append(head_weights.detach().cpu().numpy())
                    
                    # Find the correct head name from the ordered list using the camera suffix
                    head_name = next((name for name, _ in ordered_head_names if name.lower() == suffix.lower()), suffix.capitalize())
                    head_color = idx_to_color.get(relevant_heads[i], idx_to_color[-1])  # Use indigo as default if index not found
                    head_info.append((head_name, head_color))

                # Combine attention maps
                combined_map = combine_attention_maps_to_rgb(attn_maps, [color for _, color in head_info])
                combined_map_img = Image.fromarray(combined_map)

                # Resize and blend with the original image
                combined_map_img = combined_map_img.resize((900, 300))
                blended_img = Image.blend(cams[0], combined_map_img, blend_strength)

                return blended_img, head_info

            # Helper function to prepare camera images
            def prepare_camera_images(norm_rgb):
                cams = []
                for i in range(len([c for c in g_conf.DATA_USED if 'rgb' in c])):
                    rgb_img = utils.inverse_normalize(norm_rgb[-1][i],
                                                    g_conf.IMG_NORMALIZATION['mean'],
                                                    g_conf.IMG_NORMALIZATION['std'])
                    rgb_img = rgb_img.detach().cpu().numpy().squeeze(0)
                    rimg = (rgb_img.transpose(1, 2, 0) * 255).astype(np.uint8)
                    cams.append(Image.fromarray(rimg).resize((300, 300)))
                return [Image.fromarray(np.hstack([np.array(img) for img in cams]))]

            if self._model.name in ['CIL_multiview_vit_oneseq', 'CIL_multiview', 'CIL_multiview_deit_oneseq']:
                cams = []
                for i in range(len([c for c in g_conf.DATA_USED if 'rgb' in c])):
                    rgb_img = inverse_normalize(self.norm_rgb[-1][i],
                                                g_conf.IMG_NORMALIZATION['mean'],
                                                g_conf.IMG_NORMALIZATION['std'])
                    rgb_img = rgb_img.detach().cpu().numpy().squeeze(0)
                    rimg = (rgb_img.transpose(1, 2, 0) * 255).astype(np.uint8)
                    cams.append(rimg)
                cams = rearrange(cams, 'c h w C -> h (c w) C')
                cams = [Image.fromarray(cams).resize((900, 300))]
            else:
                cams = []
                for i in range(len([c for c in g_conf.DATA_USED if 'rgb' in c])):
                    rgb_img = inverse_normalize(self.norm_rgb[-1][i],
                                                g_conf.IMG_NORMALIZATION['mean'],
                                                g_conf.IMG_NORMALIZATION['std'])
                    rgb_img = rgb_img.detach().cpu().numpy().squeeze(0)
                    rimg = Image.fromarray((rgb_img.transpose(1, 2, 0) * 255).astype(np.uint8)).resize((300, 300))
                    cams.append(rimg)

            # Get the 3rd person view
            rgb_backontop = Image.fromarray(current_input_data['rgb_backontop'][1])

            # Get the command
            if g_conf.DATA_COMMAND_ONE_HOT:
                cmd = decode_directions_6(self.direction[-1].detach().cpu().numpy().squeeze(0))
            else:
                cmd = self.direction[-1].detach().cpu().numpy().squeeze(0) + 1

            command_sign_dict = {
                1.0: 'turn_left.png',
                2.0: 'turn_right.png',
                3.0: 'go_straight.png',
                4.0: 'follow_lane.png',
                5.0: 'change_left.png',
                6.0: 'change_right.png'
            }

            command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', command_sign_dict[float(cmd)]))
            command_sign = command_sign.resize((280, 70))

            # Define image characteristics
            border_height_top = 120
            border_height_bottom = 120
            images_separation_horizontally = 10  # how many pixels between images (horizontally)
            images_separation_vertically = rgb_backontop.height - 2 * cams[0].height  # how many pixels between images (vertically)

            # Background image (black)
            l = 4/3 if g_conf.CMD_SPD_TOKENS else 1
            mat_width = int(rgb_backontop.width + l * len(cams) * (cams[0].width + images_separation_horizontally)) + 75
            mat_height = int(border_height_top + rgb_backontop.height + border_height_bottom)
            mat = Image.new('RGB', (mat_width, mat_height), (0, 0, 0))
            draw_mat = ImageDraw.Draw(mat)

            # Set fonts
            font = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 30)
            font_2 = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 55)
            font_3 = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 25)

            # third person
            draw_mat.text((260, 40), "Third Person Perspective", fill=(255, 255, 255), font=font_2)
            mat.paste(rgb_backontop, (0, border_height_top))

            # Do something different according to the model name
            if self._model.name in ['CIL_multiview_vit_oneseq', 'CIL_multiview', 'CIL_multiview_deit_oneseq']:
                # Texts above each view
                if not g_conf.NO_ACT_TOKENS:
                    view_titles = [
                        'RGB Cameras Input' if g_conf.ONE_ACTION_TOKEN else '[STR] Attention Maps',
                        '[ACT] Attention Maps' if g_conf.ONE_ACTION_TOKEN else '[ACC] Attention Maps'
                    ]
                else:
                    act_output = self._model.params['action_output'].get('type', None)
                    act_query = True if (act_output is not None and 'decoder' in act_output) else False
                    view_titles = [
                        'RGB Cameras Input',
                        'ACT q Attention' if act_query else 'P2P Attention Maps'
                    ]

                if not g_conf.NO_ACT_TOKENS:  # If there are action tokens
                    if not g_conf.ONE_ACTION_TOKEN:  # If there are multiple action tokens
                        # Get the steering [STR] attention map
                        attmaps = self.attn_weights[1] if isinstance(self.attn_weights, tuple) else self.attn_weights
                        grayscale_cam_str = get_grayscale_attn_map(attmaps, -2, 900, 300, one_seq=True)
                        topl_gradcam = blend_gradcam_cameraimg(grayscale_cam_str, self.cmap_2, cams, 0, 0.5)

                        # Get the acceleration [ACC] attention map
                        grayscale_cam_acc = get_grayscale_attn_map(attmaps, -1, 900, 300, one_seq=True)
                        bottoml_gradcam = blend_gradcam_cameraimg(grayscale_cam_acc, self.cmap_2, cams, 0, 0.5)

                    else:
                        # Only one attention map, so leave the top as the input rgb image
                        topl_gradcam = [cams[0]]

                        # Get the action [ACT] attention map
                        grayscale_cam_act = get_grayscale_attn_map(self.attn_weights, -1, 900, 300, one_seq=True)
                        bottoml_gradcam = blend_gradcam_cameraimg(grayscale_cam_act, self.cmap_2, cams, 0, 0.5)

                else:
                    # Let's plot the average attention map from all patches
                    topl_gradcam = [cams[0]]

                    # Get the attention map
                    if g_conf.MHA_ATTENTION_LOSS:
                        # Multi-head attention visualization
                        bottoml_gradcam, head_info = multihead_attention_viz(self.attn_weights, cams, blend_strength=0.8)
                        bottoml_gradcam = [bottoml_gradcam]

                        # Calculate the total width of the title
                        title_width = sum(draw_mat.textsize(f"{head_name} ", font=font)[0] for head_name, _ in head_info)
                        title_width += draw_mat.textsize("Att. Maps", font=font)[0]

                        # Calculate the starting position to center the title
                        attention_map_width = 900  # Width of the attention map image
                        x_pos = rgb_backontop.width + images_separation_horizontally + (attention_map_width - title_width) // 2

                        # Draw the colored title for multi-head attention
                        for head_name, color in head_info:
                            draw_mat.text((x_pos, border_height_top + cams[0].height + 35), f"{head_name} ", font=font, fill=color)
                            x_pos += draw_mat.textsize(f"{head_name} ", font=font)[0]
                        draw_mat.text((x_pos, border_height_top + cams[0].height + 35), "Att. Maps", font=font, fill=(255, 255, 255))

                        # Replace 'P2P Attention Maps' with our new colored title
                        # view_titles[1] = colored_title

                    else:
                        grayscale_cam_patch = get_grayscale_attn_map(self.attn_weights, 0, 900, 300, one_seq=True)
                        bottoml_gradcam = blend_gradcam_cameraimg(grayscale_cam_patch, self.cmap_2, cams, 0, 0.5)

                        # Center the default title
                        title = view_titles[1]  # This will be 'P2P Attention Maps' or 'ACT q Attention'
                        title_width = draw_mat.textsize(title, font=font)[0]
                        attention_map_width = 900  # Width of the attention map image
                        x_pos = rgb_backontop.width + images_separation_horizontally + (attention_map_width - title_width) // 2
                        draw_mat.text((x_pos, border_height_top + cams[0].height + 35), title, fill=(255, 255, 255), font=font)

            else:
                # Separate the attention maps
                cam_attn_weights, steer_attn_weights, accel_attn_weights = self.attn_weights

                # Get the acceleration [ACC] attention map
                grayscale_cam_acc = cam_attn_weights[:, -1, :, :].detach().cpu().numpy()  # [S*cam, H, W]; ACC token
                # grayscale_cam_acc = grayscale_cam_acc / grayscale_cam_acc.sum(axis=(1, 2))[:, None, None]  # normalize
                grayscale_cam_acc = grayscale_cam_acc.transpose(1, 2, 0)  # [H, W, S*cam]
                grayscale_cam_acc = cv2.resize(grayscale_cam_acc, (g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2]))  # cv2 thinks it has multiple channels
                grayscale_cam_acc = grayscale_cam_acc.transpose(2, 0, 1)  # [S*cam, H, W]

                gradcams_acc = []
                for cam_id in range(len([c for c in g_conf.DATA_USED if 'rgb' in c])):
                    att = grayscale_cam_acc[cam_id, :]
                    alpha = min(0.6, accel_attn_weights.squeeze()[cam_id].item())
                    cmap_att = np.delete(self.cmap_2(att), 3, 2)
                    cmap_att = Image.fromarray((cmap_att * 255).astype(np.uint8)).resize((300, 300))
                    gcacc = Image.blend(cams[cam_id], cmap_att, alpha=alpha)
                    gradcams_acc.append(gcacc)

                # Get the steering [STR] attention map
                grayscale_cam_str = cam_attn_weights[:, -2, :, :].detach().cpu().numpy()  # [S*cam, H, W]; STR token
                # grayscale_cam_str = grayscale_cam_str / grayscale_cam_str.sum(axis=(1, 2))[:, None, None]  # normalize
                grayscale_cam_str = grayscale_cam_str.transpose(1, 2, 0)  # [H, W, S*cam]
                grayscale_cam_str = cv2.resize(grayscale_cam_str, (g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2]))  # cv2 thinks it has multiple channels
                grayscale_cam_str = grayscale_cam_str.transpose(2, 0, 1)  # [S*cam, H, W]

                gradcams_str = []
                for cam_id in range(len([c for c in g_conf.DATA_USED if 'rgb' in c])):
                    att = grayscale_cam_str[cam_id, :]
                    alpha = min(0.6, steer_attn_weights.squeeze()[cam_id].item())
                    cmap_att = np.delete(self.cmap_2(att), 3, 2)
                    cmap_att = Image.fromarray((cmap_att * 255).astype(np.uint8)).resize((300, 300))
                    gcstr = Image.blend(cams[cam_id], cmap_att, alpha=alpha)
                    gradcams_str.append(gcstr)

            # Attention Maps
            draw_mat.text((330 + rgb_backontop.width, 80), view_titles[0], fill=(255, 255, 255), font=font)
            if self._model.name in ['CIL_multiview_vit_oneseq', 'CIL_multiview', 'CIL_multiview_deit_oneseq']:
                mat.paste(topl_gradcam[0], (rgb_backontop.width + images_separation_horizontally, border_height_top))
            else:
                mat.paste(gradcams_str[0], (rgb_backontop.width + images_separation_horizontally, border_height_top))
                mat.paste(gradcams_str[1], (rgb_backontop.width + 2*images_separation_horizontally + int(cams[0].width),
                                            border_height_top))
                mat.paste(gradcams_str[2], (rgb_backontop.width + 3*images_separation_horizontally + 2*int(cams[0].width),
                                            border_height_top))

            # if not g_conf.MHA_ATTENTION_LOSS:
            #     draw_mat.text((330 + rgb_backontop.width, border_height_top + cams[0].height + 35), view_titles[1],
            #                 fill=(255, 255, 255), font=font)
            if self._model.name in ['CIL_multiview_vit_oneseq', 'CIL_multiview', 'CIL_multiview_deit_oneseq']:
                mat.paste(bottoml_gradcam[0], (rgb_backontop.width + images_separation_horizontally,
                                               border_height_top + cams[0].height + images_separation_vertically))
            else:
                mat.paste(gradcams_acc[0], (rgb_backontop.width + images_separation_horizontally,
                                            border_height_top + cams[0].height + images_separation_vertically))
                mat.paste(gradcams_acc[1], (rgb_backontop.width + 2*images_separation_horizontally + int(cams[0].width),
                                            border_height_top + cams[0].height + images_separation_vertically))
                mat.paste(gradcams_acc[2], (rgb_backontop.width + 3*images_separation_horizontally + 2*int(cams[0].width),
                                            border_height_top + cams[0].height + images_separation_vertically))

            if g_conf.CMD_SPD_TOKENS:
                # Get the token-to-token attention map
                grayscale_cam_t2t = get_grayscale_attn_map(self.attn_weights[0], resize_width=375, resize_height=300)
                cmap_att = np.delete(self.cmap_2(grayscale_cam_t2t), 3, 3)[0]
                cmap_att = Image.fromarray((cmap_att * 255).astype(np.uint8))
                # Paste them
                title = '[CMD] [SPD] [STR] [ACC] Patches' if not g_conf.ONE_ACTION_TOKEN else '[CMD] [SPD] [ACT] Patches'
                draw_mat.text((rgb_backontop.width + 2*images_separation_horizontally + cams[0].width, 80), title,
                              fill=(255, 255, 255), font=font_3)
                mat.paste(cmap_att, (rgb_backontop.width + 2*images_separation_horizontally + cams[0].width, border_height_top))
            # command
            draw_mat.text((rgb_backontop.width, border_height_top + rgb_backontop.height + 15),
                          "Command", fill=(255, 255, 255), font=font)
            draw_mat.text((rgb_backontop.width+25, 120 + rgb_backontop.height + 45),
                          "Input", fill=(255, 255, 255), font=font)
            mat.paste(command_sign, (120 + rgb_backontop.width + 35, border_height_top + rgb_backontop.height + 30))

            # speed
            draw_mat.text((140 + rgb_backontop.width + command_sign.width + 110,
                           border_height_top + rgb_backontop.height + 15), "Speed (m/s)",
                          fill=(255, 255, 255), font=font)
            draw_mat.text((140 + rgb_backontop.width + command_sign.width + 120,
                           border_height_top + rgb_backontop.height + 45), "Input",
                          fill=(255, 255, 255), font=font)
            speed_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                gauge={'axis': {'range': [0, 12], 'visible': False},
                       'bordercolor': "white",
                       'borderwidth': 3},
                value=round(current_input_data['SPEED'][1]['speed'], 3),
                domain={'x': [0, 1], 'y': [0, 1]},))

            speed_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial", "size": 3})
            buf = io.BytesIO()
            speed_gauge.write_image(buf, format='png')
            buf.seek(0)
            speed_gauge = Image.open(buf)
            speed_gauge = speed_gauge.resize((202, 144))
            speed_gauge = speed_gauge.crop((20, 20, 182, 124))
            mat.paste(speed_gauge, (140 + rgb_backontop.width + command_sign.width + 280, border_height_top + rgb_backontop.height + 5))
            draw_mat.text((1760, border_height_top + rgb_backontop.height + 75), "0", fill=(255, 255, 255), font=font_3)
            draw_mat.text((1955, border_height_top + rgb_backontop.height + 75), "12", fill=(255, 255, 255), font=font_3)

            # steer
            draw_mat.text((60, border_height_top + rgb_backontop.height + 15),
                          f"Steering Prediction {np.clip(self.steer, -1, 1):.3f}",
                          fill=(255, 255, 255),
                          font=font)
            if self.steer > 0.0:
                step = [
                    {'range': [-1, 0], 'color': "black"},
                    {'range': [0, self.steer], 'color': "yellow"},
                    {'range': [self.steer, 1], 'color': "black"}]
            else:
                step = [
                    {'range': [-1, self.steer], 'color': "black"},
                    {'range': [self.steer, 0], 'color': "yellow"},
                    {'range': [0, 1], 'color': "black"}]

            steer_gauge = go.Figure(go.Indicator(
                mode="number+gauge",
                gauge={'shape': "bullet",
                       'axis': {'range': [-1, 1], 'visible': False},
                       'bordercolor': "white",
                       'borderwidth': 3,
                       'steps': step,
                       'threshold': {
                           'line': {'color': "white", 'width': 3},
                           'thickness': 1.0, 'value': 0}},
                domain={'x': [0, 1], 'y': [0, 1]}))
            steer_gauge.update_layout(
                height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial", "size": 3})
            buf = io.BytesIO()
            steer_gauge.write_image(buf, format='png')
            buf.seek(0)
            steer_gauge = Image.open(buf)
            steer_gauge = steer_gauge.resize((359, 128))
            steer_gauge = steer_gauge.crop((40, 40, 250, 100))
            mat.paste(steer_gauge, (110, border_height_top + rgb_backontop.height + 50))
            draw_mat.text((65, border_height_top + rgb_backontop.height + 60), "-1", fill=(255, 255, 255), font=font)
            draw_mat.text((340, border_height_top + rgb_backontop.height + 60),
                          "1", fill=(255, 255, 255), font=font)

            # acceleration
            draw_mat.text((550, border_height_top + rgb_backontop.height + 15),
                          f"Acceleration Prediction {np.clip(self.acceleration, -1, 1):.3f}",
                          fill=(255, 255, 255),
                          font=font)
            if self.acceleration > 0.0:
                step = [
                    {'range': [-1, 0], 'color': "black"},
                    {'range': [0, self.acceleration], 'color': "orange"},
                    {'range': [self.acceleration, 1], 'color': "black"}]
            else:
                step = [
                    {'range': [-1, self.acceleration], 'color': "black"},
                    {'range': [self.acceleration, 0], 'color': "lightgray"},
                    {'range': [0, 1], 'color': "black"}]
            acc_gauge = go.Figure(go.Indicator(
                mode="number+gauge",
                gauge={'shape': "bullet",
                       'axis': {'range': [-1, 1], 'visible': False},
                       'bordercolor': "white",
                       'borderwidth': 3,
                       'steps': step,
                       'threshold': {
                           'line': {'color': "white", 'width': 3},
                           'thickness': 1.0, 'value': 0}},
                domain={'x': [0, 1], 'y': [0, 1]}))
            acc_gauge.update_layout(
                height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial", "size": 3})
            buf = io.BytesIO()
            acc_gauge.write_image(buf, format='png')
            buf.seek(0)
            acc_gauge = Image.open(buf)
            acc_gauge = acc_gauge.resize((359, 128))
            acc_gauge = acc_gauge.crop((40, 40, 250, 100))
            mat.paste(acc_gauge, (625, border_height_top + rgb_backontop.height + 50))
            draw_mat.text((580, border_height_top + rgb_backontop.height + 60), "-1", fill=(255, 255, 255), font=font)
            draw_mat.text((855, border_height_top + rgb_backontop.height + 60), "1", fill=(255, 255, 255), font=font)
            
            if not os.path.exists(self.vision_save_path):
                os.makedirs(self.vision_save_path)
            # mat = mat.resize((int(mat.size[0] / 2), int(mat.size[1] / 2)))   # comment this if you don't want to resize
            mat.save(os.path.join(self.vision_save_path, f'{self.datapoint_count:06d}.jpg'))