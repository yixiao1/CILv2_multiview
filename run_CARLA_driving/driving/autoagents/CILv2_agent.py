#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
import os
import io
import torch
import json
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from driving.utils.route_manipulation import downsample_route
from driving.envs.sensor_interface import SensorInterface

from configs import g_conf, merge_with_yaml, set_type_of_process
from network.models_console import Models
from _utils.training_utils import DataParallelWrapper
from dataloaders.transforms import encode_directions_4, encode_directions_6, inverse_normalize, decode_directions_4, \
    decode_directions_6
from driving.utils.waypointer import Waypointer
from driving.utils.route_manipulation import interpolate_trajectory

from omegaconf import OmegaConf
from network.models.architectures.Roach_rl_birdview.birdview.chauffeurnet import ObsManager
from network.models.architectures.Roach_rl_birdview.utils.traffic_light import TrafficLightHandler
from importlib import import_module


def checkpoint_parse_configuration_file(filename):
    with open(filename, 'r') as f:
        configuration_dict = json.loads(f.read())

    return configuration_dict['yaml'], configuration_dict['checkpoint'], \
           configuration_dict['agent_name'], configuration_dict['lens_circle_set']

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

def get_entry_point():
    return 'CILv2_agent'


class Track(Enum):
    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'


class CILv2_agent(object):
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
        self.norm_rgb = None
        self.norm_speed = None
        self.checkpoint = None
        self.world = None
        self.map = None

        # agent's initialization
        self.setup_model(path_to_conf_file)

        self.cmap_2 = plt.get_cmap('inferno')
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
        
        self._model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        if torch.cuda.device_count() > 1 and g_conf.DATA_PARALLEL:
            print("Using multiple GPUs parallel! ")
            print(torch.cuda.device_count(), 'GPUs to be used: ', os.environ["CUDA_VISIBLE_DEVICES"])
            self._model = DataParallelWrapper(self._model)
        self.checkpoint = torch.load(os.path.join(exp_dir, 'checkpoints', f'{self._model.name}_{checkpoint_number}.pth'))
        print(self._model.name + '_' + str(checkpoint_number) + '.pth', "loaded from ",
              os.path.join(exp_dir, 'checkpoints'))
        if isinstance(self._model, torch.nn.DataParallel):
            self._model.module.load_state_dict(self.checkpoint['model'])
        else:
            self._model.load_state_dict(self.checkpoint['model'])
        self._model.cuda()
        self._model.eval()

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

        if self.vision_save_path:
            sensors = [
                {'type': 'sensor.camera.rgb', 'x': -4.5, 'y': 0.0, 'z': 4.0, 'roll': 0.0, 'pitch': -20.0, 'yaw': 0.0,
                 'width': 1088, 'height': 680, 'fov': 120, 'id': 'rgb_backontop', 'lens_circle_setting': False},

                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_central', 'lens_circle_setting': self.lens_circle_setting},

                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_left', 'lens_circle_setting': self.lens_circle_setting},

                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_right', 'lens_circle_setting': self.lens_circle_setting},

                {'type': 'sensor.other.gnss', 'id': 'GPS'},

                {'type': 'sensor.other.imu', 'id': 'IMU'},

                {'type': 'sensor.speedometer', 'id': 'SPEED'},

                {'type': 'sensor.can_bus', 'id': 'can_bus'}
            ]

        else:
            sensors = [
                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_central', 'lens_circle_setting': self.lens_circle_setting},

                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_left', 'lens_circle_setting': self.lens_circle_setting},

                {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                 'width': 300, 'height': 300, 'fov': 60, 'id': 'rgb_right', 'lens_circle_setting': self.lens_circle_setting},

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
        self.norm_rgb = [[self.process_image(self.input_data[camera_type][1]).unsqueeze(0).cuda() for camera_type in g_conf.DATA_USED]]
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

        actions_outputs, self.attn_weights = self._model.forward_eval(self.norm_rgb, self.direction, self.norm_speed)

        # actions_outputs = self._model.forward_eval(self.norm_rgb, self.direction, self.norm_speed)

        last_action_outputs = self.process_control_outputs(actions_outputs.detach().cpu().numpy().squeeze())

        self.steer, self.throttle, self.brake = last_action_outputs
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
        if self.vision_save_path:
            cams = []
            for i in range(len(g_conf.DATA_USED)):
                rgb_img = inverse_normalize(self.norm_rgb[-1][i], g_conf.IMG_NORMALIZATION['mean'], g_conf.IMG_NORMALIZATION['std']).detach().cpu().numpy().squeeze(0)
                cams.append(Image.fromarray((rgb_img.transpose(1, 2, 0) * 255).astype(np.uint8)))

            # target_layers = [self._model._model.encoder_embedding_perception.encoder.layers.encoder_layer_11]
            # cam = GradCAM(model=self._model._model, target_layers=target_layers)

            # input_tensor_list = [self.norm_rgb, self.direction, self.norm_speed]  # [3, 3, H, W], [no_commands], [1]
            # with torch.enable_grad():
            #     grayscale_cam = cam(input_tensor_list=input_tensor_list)   # [S*cam, H, W]

            grayscale_cam = self.attn_weights[:, 0, :, :].detach().cpu().numpy()  # [S*cam, H, W]; just the ACC token
            grayscale_cam = grayscale_cam.transpose(1, 2, 0)  # [H, W, S*cam]
            grayscale_cam = cv2.resize(grayscale_cam, (g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2]),
                                       interpolation=cv2.INTER_AREA)  # cv2 thinks it has multiple channels
            grayscale_cam = grayscale_cam.transpose(2, 0, 1)  # [S*cam, H, W]

            gradcams = []
            for cam_id in range(len(g_conf.DATA_USED)):
                att = grayscale_cam[cam_id, :]
                cmap_att = np.delete(self.cmap_2(att), 3, 2)
                cmap_att = Image.fromarray((cmap_att * 255).astype(np.uint8))
                gradcams.append(Image.blend(cams[cam_id], cmap_att, 0.5))

            # last_input_ontop = Image.fromarray(current_input_data['rgb_ontop'][1])
            rgb_backontop = Image.fromarray(current_input_data['rgb_backontop'][1])

            if g_conf.DATA_COMMAND_ONE_HOT:
                cmd = decode_directions_6(self.direction[-1].detach().cpu().numpy().squeeze(0))
            else:
                cmd = self.direction[-1].detach().cpu().numpy().squeeze(0) + 1
            if float(cmd) == 1.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'turn_left.png'))

            elif float(cmd) == 2.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'turn_right.png'))

            elif float(cmd) == 3.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'go_straight.png'))

            elif float(cmd) == 4.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'follow_lane.png'))

            elif float(cmd) == 5.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'change_left.png'))

            elif float(cmd) == 6.0:
                command_sign = Image.open(os.path.join(os.getcwd(), 'signs', '6_directions', 'change_right.png'))

            else:
                raise RuntimeError()

            command_sign = command_sign.resize((280, 70))

            mat = Image.new('RGB', (rgb_backontop.width+len(cams)*(cams[0].width + 10),
                                    120 + rgb_backontop.height+120), (0, 0, 0))
            draw_mat = ImageDraw.Draw(mat)
            font = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 30)
            font_2 = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 55)
            font_3 = ImageFont.truetype(os.path.join(os.getcwd(), 'signs', 'arial.ttf'), 25)

            # third person
            draw_mat.text((260, 40), str("Third Person Perspective"), fill=(255, 255, 255), font=font_2)
            mat.paste(rgb_backontop, (0, 120))

            # RGB
            draw_mat.text((330 + rgb_backontop.width, 30), str("RGB Cameras Input"), fill=(255, 255, 255), font=font)
            draw_mat.text((120 + rgb_backontop.width, 80), str("-60.0"+'\u00b0'), fill=(255, 255, 255), font=font)
            draw_mat.text((435 + rgb_backontop.width, 80), str("0.0"+'\u00b0'), fill=(255, 255, 255), font=font)
            draw_mat.text((735 + rgb_backontop.width, 80), str("60.0"+'\u00b0'), fill=(255, 255, 255), font=font)
            mat.paste(cams[0], (rgb_backontop.width+10, 120))
            mat.paste(cams[1], (rgb_backontop.width+10 + int(cams[0].width) + 10, 120))
            mat.paste(cams[2], (rgb_backontop.width+10 + int((cams[0].width) + 10)*2, 120))

            # activation maps
            draw_mat.text((360 + rgb_backontop.width, 120 + cams[0].height + 35), str("[ACC] Attention Maps"),
                          fill=(255, 255, 255), font=font)
            mat.paste(gradcams[0], (rgb_backontop.width+10, 120+cams[0].height+80))
            mat.paste(gradcams[1], (rgb_backontop.width+10 + int(cams[0].width) + 10, 120+cams[0].height+80))
            mat.paste(gradcams[2], (rgb_backontop.width+10 + int((cams[0].width) + 10)*2,  120+cams[0].height+80))

            # command
            draw_mat.text((rgb_backontop.width, 120 + rgb_backontop.height + 15),
                          str("Command"),
                          fill=(255, 255, 255),
                          font=font)
            draw_mat.text((rgb_backontop.width+25, 120 + rgb_backontop.height + 45),
                          str("Input"),
                          fill=(255, 255, 255),
                          font=font)
            mat.paste(command_sign, (120 + rgb_backontop.width + 35, 120+rgb_backontop.height + 30))

            # speed
            draw_mat.text((140 + rgb_backontop.width + command_sign.width + 110, 120+rgb_backontop.height + 15), str("Speed " + "(m/s)"),
                          fill=(255, 255, 255), font=font)
            draw_mat.text((140 + rgb_backontop.width + command_sign.width + 120, 120+rgb_backontop.height + 45), str("Input "),
                          fill=(255, 255, 255), font=font)
            speed_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                gauge={'axis': {'range': [0, 12], 'visible': False},
                       'bordercolor': "white",
                       'borderwidth': 3},
                value= round(current_input_data['SPEED'][1]['speed'], 3),
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
            mat.paste(speed_gauge, (140 + rgb_backontop.width + command_sign.width + 280, 120+rgb_backontop.height+5))
            draw_mat.text((1760, 120 + rgb_backontop.height + 75), str("0" ), fill=(255, 255, 255), font=font_3)
            draw_mat.text((1955, 120 + rgb_backontop.height + 75),
                          str("12"), fill=(255, 255, 255), font=font_3)

            # steer
            draw_mat.text((60, 120 + rgb_backontop.height + 15),
                          str("Steering Prediction  "+ "%.3f" % np.clip(self.steer, -1, 1)),
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
                           'thickness': 1.0, 'value': 0},},
                domain={'x': [0, 1], 'y': [0, 1]}))
            steer_gauge.update_layout(
                height = 250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial", "size": 3})
            buf = io.BytesIO()
            steer_gauge.write_image(buf, format='png')
            buf.seek(0)
            steer_gauge = Image.open(buf)
            steer_gauge = steer_gauge.resize((359, 128))
            steer_gauge = steer_gauge.crop((40, 40, 250, 100))
            mat.paste(steer_gauge, (110 , 120+rgb_backontop.height + 50))
            draw_mat.text((65, 120 + rgb_backontop.height + 60), str("-1" ), fill=(255, 255, 255), font=font)
            draw_mat.text((340, 120 + rgb_backontop.height + 60),
                          str("1"), fill=(255, 255, 255), font=font)

            # acceleration
            draw_mat.text((550 , 120 + rgb_backontop.height + 15),
                          str("Acceleration Prediction  "+ "%.3f" % np.clip(self.acceleration, -1, 1)),
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
                           'thickness': 1.0, 'value': 0},},
                domain={'x': [0, 1], 'y': [0, 1]}))
            acc_gauge.update_layout(
                height = 250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial", "size": 3})
            buf = io.BytesIO()
            acc_gauge.write_image(buf, format='png')
            buf.seek(0)
            acc_gauge = Image.open(buf)
            acc_gauge = acc_gauge.resize((359, 128))
            acc_gauge = acc_gauge.crop((40, 40, 250, 100))
            mat.paste(acc_gauge, (625 , 120+rgb_backontop.height + 50))
            draw_mat.text((580, 120 + rgb_backontop.height + 60), str("-1" ), fill=(255, 255, 255), font=font)
            draw_mat.text((855, 120 + rgb_backontop.height + 60), str("1"), fill=(255, 255, 255), font=font)
            
            if not os.path.exists(self.vision_save_path):
                os.makedirs(self.vision_save_path)
            if self.datapoint_count % self.save_frequence == 0:
                mat = mat.resize((int(mat.size[0] / 2), int(mat.size[1] / 2)))   # comment this if you don't want to resize
                mat.save(os.path.join(self.vision_save_path, str(self.datapoint_count).zfill(6) + '.jpg'))