from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from enum import Enum

import carla
import re
import torch
import json

import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib.pyplot as plt
from _utils import utils

from driving.utils.route_manipulation import downsample_route
from driving.envs.sensor_interface import SensorInterface

from configs import g_conf
from dataloaders.transforms import encode_directions_4, encode_directions_6, get_virtual_noise_from_depth
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
    return 'B2D_agent'


class Track(Enum):
    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'


class B2D_agent(object):
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
        # TODO
        pass

    def setup_expert_agent(self, path_to_conf_file):
        # TODO
        pass

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

    def sensors(self):
        """
        Define the sensor suite required by the agent
        
        :return: a list containing the required sensors
        """

        sensors = [
            # RGB cameras (6 cameras for 360° coverage)
            {
                'type': 'sensor.camera.rgb',
                'x': 0.80, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_RIGHT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -2.0, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'CAM_BACK'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_RIGHT'
            },
            
            # LiDAR
            {
                'type': 'sensor.lidar.ray_cast',
                'x': -0.39, 'y': 0.0, 'z': 1.84,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'range': 85,
                'rotation_frequency': 10,
                'channels': 64,
                'points_per_second': 600000,
                'dropoff_general_rate': 0.0,
                'dropoff_intensity_limit': 0.0,
                'dropoff_zero_intensity': 0.0,
                'id': 'LIDAR_TOP'
            },

            # GPS
            {'type': 'sensor.other.gnss', 'id': 'GPS'},
            
            # IMU
            {
                'type': 'sensor.other.imu',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'IMU'
            },
            
            # Speedometer
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'SPEED'},
        ]

        return sensors

    def __call__(self, timestamp):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        self.input_data = self.sensor_interface.get_data()
        
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
        # TODO
        pass