# Create a .mp4 video of the collected data; the user must indicate the name of the sensors to be displayed,
# as well as the name of the corresponding *.json file for each tick of the video. The video will be saved in
# ./out/data_analysis/DATASET_NAME/VIDEO_NAME.mp4

# Each .json file has the following structure:
# {
#     "acceleration": 1.0,
#     "brake": 0.0,
#     "direction": 4.0,
#     "ego_location": [
#         262.61981201171875,
#         330.6069030761719,
#         0.02722419612109661
#     ],
#     "ego_rotation": [
#         0.0032716605346649885,
#         0.006111423019319773,
#         -0.0030822751577943563
#     ],
#     "hand_brake": false,
#     "navigate_wp_location": [
#         263.6197814941406,
#         330.6070861816406,
#         0.0
#     ],
#     "navigate_wp_rotation": [
#         0.0,
#         0.006118634715676308,
#         0.0
#     ],
#     "reverse": false,
#     "speed": 2.892015407215783e-05,
#     "steer": 0.06136000156402588,
#     "throttle": 1.0
#  }

# Thus, we are only interested in the speed, acceleration, and direction values.


import os
import sys
import json
import click
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np
from glob import glob


def get_data_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def get_speed(data):
    return data['speed']

def get_acceleration(data):
    return data['acceleration']

def get_direction(data):
    return data['direction']


# The directory structure is as follows:
# /ROOT
# -> weather_%05d, the number idincating the route
# -> -> can_bus%06d.json
# -> -> rgb_central%06d.png
# -> -> rgb_left%06d.png
# -> -> rgb_right%06d.png
# -> -> depth_central%06d.png
# -> -> depth_left%06d.png
# -> -> depth_right%06d.png
# -> -> ss_central%06d.png
# -> -> ss_left%06d.png
# -> -> ss_right%06d.png

def get_data_from_dir(dir_path, sensor_name):
    data = []
    for file in os.listdir(dir_path):
        if file.endswith(".json"):
            data.append(get_data_from_json(os.path.join(dir_path, file)))
    if sensor_name == 'speed':
        return list(map(get_speed, data))
    elif sensor_name == 'acceleration':
        return list(map(get_acceleration, data))
    elif sensor_name == 'direction':
        return list(map(get_direction, data))
    else:
        raise ValueError("Invalid sensor name: {}".format(sensor_name))

# For each route, let's get all of the datafiles:
def get_data_paths(root_path, sensor_name: list):
    """ Use glob to get them all, then order them. """
    data_paths = []
    for sensor in sensor_name:
        data_paths.append(glob(os.path.join(root_path, f'{sensor}*')))
    # Let's sort the data_paths by the number of each file, which is at the end:
    for i in range(len(data_paths)):
        data_paths[i].sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    return data_paths