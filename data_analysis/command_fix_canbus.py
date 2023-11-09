# Fix the command in every can_bus%06d.json file in the dataset
import glob
import os
from tqdm import tqdm
import click
import json
import shutil
from typing import Union
from _utils.utils import sort_nicely
import math
from multiprocessing import Pool


def process_container(args) -> type(None):
    container_path, dataset_path = args
    container = container_path.split(os.sep)[-1]

    json_path_list = glob.glob(os.path.join(container_path, 'can_bus*.json'))
    sort_nicely(json_path_list)
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


@click.command()
@click.option('--dataset-path', help='Path to the root of your dataset to modify', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
def main(dataset_path: Union[str, os.PathLike]):
    """ Manually fix a bug in the dataset wherein the command/direction is given too soon to the ego vehicle. """
    all_containers_path_list = glob.glob(os.path.join(dataset_path, '*'))
    sort_nicely(all_containers_path_list)

    args = [(container_path, dataset_path) for container_path in all_containers_path_list]

    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm(pool.imap(process_container, args), total=len(all_containers_path_list)):
            pass

if __name__ == '__main__':
    main()
