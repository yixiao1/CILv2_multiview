import os
import yaml
import numpy as np
import json

from yaml.loader import SafeLoader

def get_env_config_from_train_test_config(experiment_name, config_name):
    experiment_path = f"{os.getenv('HOME')}/results/rlad2/{experiment_name}"
    train_test_config = get_config(f'{experiment_path}/configs/{config_name}')
    env_config_name = train_test_config['env_name']
    env_config = get_config(f"{experiment_path}/configs/{env_config_name}")
    return env_config

def get_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

def get_config_with_env_var(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    content = os.path.expandvars(content)
    return yaml.safe_load(content)

def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f)

def convert_numpy_dict_to_list_dict(in_dict):
    final_dict = {}
    for key, value in in_dict.items():
        if isinstance(value, np.ndarray):
            value_ = value.tolist()
        else:
            value_ = value

        final_dict[key] = value_
    return final_dict

def convert_list_dict_to_numpy_dict(in_dict):
    final_dict = {}
    for key, value in in_dict.items():
        if isinstance(value, list):
            value_ = np.asarray(value)
        else:
            value_ = value

        final_dict[key] = value_
    return final_dict

def create_empty_file(filename):
    with open(filename, 'w') as file:
        pass

def save_json(filename, json_file):
    with open(filename, "w") as f:
        json.dump(json_file, f, indent=4)