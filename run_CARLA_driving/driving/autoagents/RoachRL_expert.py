#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import os
import carla
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import copy
from importlib import import_module

from driving.utils.route_manipulation import downsample_route
from driving.envs.sensor_interface import SensorInterface
from driving.utils.route_manipulation import interpolate_trajectory

from driving.utils.waypointer import Waypointer
from omegaconf import OmegaConf
from network.models.architectures.Roach_rl_birdview.birdview.chauffeurnet import ObsManager
import network.models.architectures.Roach_rl_birdview.utils.transforms as trans_utils
from network.models.architectures.Roach_rl_birdview.utils.traffic_light import TrafficLightHandler


def checkpoint_parse_configuration_file(filename):

    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name']

def get_entry_point():
    return 'RoachRL_expert'

class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

class RoachRL_expert(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file, save_driving_vision):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
        self.waypointer = None
        self.vision_save_path = save_driving_vision

        # agent's initialization
        self.setup_model(path_to_conf_file)

        self.cmap_2 = plt.get_cmap('jet')
        self.datapoint_count = 0
        self.save_frequence = 1

    def setup_model(self, path_to_conf_file):
        exp_dir = os.path.join('/', os.path.join(*path_to_conf_file.split('/')[:-1]))
        yaml_conf, checkpoint_number, _ = checkpoint_parse_configuration_file(path_to_conf_file)
        cfg = OmegaConf.load(os.path.join(exp_dir, yaml_conf))

        self._ckpt = os.path.join(exp_dir, 'checkpoints', str(checkpoint_number) + '.pth')
        cfg = OmegaConf.to_container(cfg)

        self._obs_configs = cfg['obs_configs']
        self._train_cfg = cfg['training']

        # prepare policy
        self._policy_class = load_entry_point(cfg['policy']['entry_point'])
        self._policy_kwargs = cfg['policy']['kwargs']
        print(f'Loading checkpoint: {self._ckpt}')
        self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
        self._policy = self._policy.eval()

        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

        self._obs_managers= ObsManager(cfg['obs_configs']['birdview'])

    def set_world(self, world):
        self.world=world
        self.map=self.world.get_map()
        TrafficLightHandler.reset(self.world)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in
                                         ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._route_plan = global_plan_world_coord
        self.waypointer = Waypointer(self.world, global_plan_gps = self._global_plan, global_route=global_plan_world_coord)

    def reset_global_plan(self):
        """
        reset the plan (route) for the agent
        """
        current_loc = self._ego_vehicle.get_location()
        last_gps, _ = self._global_plan[-1]
        last_loc = self.waypointer.gps_to_location([last_gps['lat'], last_gps['lon'], last_gps['z']])
        gps_route, route = interpolate_trajectory(self.world, [current_loc, last_loc])

        self._route_plan = route
        self._obs_managers.attach_ego_vehicle(self._ego_vehicle, self._route_plan)

        ds_ids = downsample_route(route, 50)
        self.route = [(route[x][0], route[x][1]) for x in ds_ids]
        self._global_plan = [gps_route[x] for x in ds_ids]

        self.waypointer.reset_route(global_plan_gps=self._global_plan, global_route=route)
        return route

    def set_ego_vehicle(self, ego_vehicle):
        self._ego_vehicle=ego_vehicle
        self._obs_managers.attach_ego_vehicle(self._ego_vehicle, self._route_plan)


    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        """

        sensors = [
            # RGB cameras
            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_central', 'lens_circle_setting': False},

            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_left', 'lens_circle_setting': False},

            {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_right', 'lens_circle_setting': False},

            # depth cameras
            {'type': 'sensor.camera.depth', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'depth_central', 'lens_circle_setting': False},

            {'type': 'sensor.camera.depth', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'depth_left', 'lens_circle_setting': False},

            {'type': 'sensor.camera.depth', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
             'width': 300, 'height': 300, 'fov': 60, 'id': 'depth_right', 'lens_circle_setting': False},

            # LiDAR cameras
            # {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #  'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_central', 'lens_circle_setting': False},
            #
            # {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
            #  'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_left', 'lens_circle_setting': False},
            #
            # {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
            #  'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_right', 'lens_circle_setting': False},

            # Radar sensors
            # {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #  'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_central', 'lens_circle_setting': False},
            #
            # {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
            #  'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_left', 'lens_circle_setting': False},
            #
            # {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
            #  'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_right', 'lens_circle_setting': False},

            {'type': 'sensor.other.gnss', 'id': 'GPS'},

            {'type': 'sensor.other.imu', 'id': 'IMU'},

            {'type': 'sensor.speedometer', 'id': 'SPEED'},

            {'type': 'sensor.can_bus', 'id': 'can_bus'}
        ]

        self.to_save_sensor_tags = ['rgb_central', 'rgb_left', 'rgb_right',
                                    'depth_central', 'depth_left', 'depth_right']
                                    # 'depth_right']

        return sensors


    def __call__(self, timestamp):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """

        self.input_data = self.sensor_interface.get_data()
        self.input_data = self.adding_roach_data(self.input_data)

        control = self.run_step()
        control.manual_gear_shift = False

        return control

    def run_step(self):
        """
        Execute one step of navigation.
        :return: control
        """

        input_data = copy.deepcopy(self.input_data)

        policy_input = self._wrapper_class.process_obs(input_data, self._wrapper_kwargs['input_states'], train=False)

        actions, _, _, _, _, _ = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)
        control = self._wrapper_class.process_act(actions, self._wrapper_kwargs['acc_as_action'], train=False)

        steer = control.steer
        throttle = control.throttle
        brake = control.brake
        cmd = self.waypointer.tick_roach(self.input_data['GPS'][1], self.input_data['IMU'][1])['command'][0]

        if not os.path.exists(self.vision_save_path):
            os.makedirs(self.vision_save_path)

        for sensor_type in self.to_save_sensor_tags:
            try:
                Image.fromarray(self.input_data[sensor_type][1], mode='RGB').save(os.path.join(self.vision_save_path,
                                                                                   sensor_type+str(self.datapoint_count).zfill(6) + '.png'))
            except:
                raise RuntimeError('Sensor type not found!')

        acc = -1 * brake if throttle == 0.0 else throttle

        # we record the driving measurement data
        data = input_data['can_bus'][1]
        data.update({'steer': np.nan_to_num(steer)})
        data.update({'throttle': np.nan_to_num(throttle)})
        data.update({'acceleration': np.nan_to_num(acc)})
        data.update({'brake': np.nan_to_num(brake)})
        data.update({'hand_brake': control.hand_brake})
        data.update({'reverse': control.reverse})
        data.update({'speed': input_data['SPEED'][1]['speed']})
        data.update({'direction': float(cmd)})
        with open(os.path.join(self.vision_save_path, 'can_bus' + str(self.datapoint_count).zfill(6) + '.json'), 'w') as fo:
            jsonObj = {}
            jsonObj.update(data)
            fo.seek(0)
            fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))
            fo.close()

        self.datapoint_count+=1

        return control

    def adding_roach_data(self, input_dict):
        obs_dict = self._obs_managers.get_observation()
        input_dict.update({'birdview': obs_dict})

        control = self._ego_vehicle.get_control()
        speed_limit = self._ego_vehicle.get_speed_limit() / 3.6 * 0.8
        control_obs = {
            'throttle': np.array([control.throttle], dtype=np.float32),
            'steer': np.array([control.steer], dtype=np.float32),
            'brake': np.array([control.brake], dtype=np.float32),
            'gear': np.array([control.gear], dtype=np.float32),
            'speed_limit': np.array([speed_limit], dtype=np.float32),
        }

        ev_transform = self._ego_vehicle.get_transform()
        acc_w = self._ego_vehicle.get_acceleration()
        vel_w = self._ego_vehicle.get_velocity()
        ang_w = self._ego_vehicle.get_angular_velocity()

        acc_ev = trans_utils.vec_global_to_ref(acc_w, ev_transform.rotation)
        vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)

        velocity_obs = {
            'acc_xy': np.array([acc_ev.x, acc_ev.y], dtype=np.float32),
            'vel_xy': np.array([vel_ev.x, vel_ev.y], dtype=np.float32),
            'vel_ang_z': np.array([ang_w.z], dtype=np.float32)
        }

        input_dict.update({'control': control_obs})
        input_dict.update({'velocity': velocity_obs})

        return input_dict

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        self._model = None
        self.checkpoint=None
        self.world=None
        self.map=None
        self._obs_managers.clean()

        self.reset()

    def reset(self):
        self.track = Track.SENSORS
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = None
        self.waypointer = None
        self.vision_save_path = None
        self.datapoint_count = 0
