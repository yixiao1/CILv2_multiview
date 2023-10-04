import os
import shutil
import json
import yaml
import numpy as np
import csv

from datetime import datetime
from yaml.loader import SafeLoader

from utilities.visualization import save_video_from_images, save_episode_plot
from utilities.configs import save_config, convert_numpy_dict_to_list_dict, create_empty_file, get_config
from utilities.common import recursive_format, format_number


class LogFile():
    def __init__(self, experiment_path, observation_config, agent_config, train_test_config, train_test_config_name, overwrite, fps, resume_training, log_every):

        self.train = True if train_test_config_name == 'train.yaml' else False
        self.train_test_config_name = train_test_config_name
        self.train_test_name = train_test_config_name.split('.')[0]
        self.test_folder_path = f"{experiment_path}/{self.train_test_name}"
        
        if self.train:
            if os.path.exists(experiment_path) and not overwrite:
                print(f'{experiment_path} already exits. ')
                raise Exception(
                    'Experiment name already exists. If you want to overwrite, use flag -ow')
            if os.path.exists(experiment_path):
                shutil.rmtree(experiment_path)
        else:
            if not os.path.exists(experiment_path):
                print(f'{experiment_path} does not exits! ')
                raise Exception('Make sure the experiment exists.')

            if os.path.exists(f"{self.test_folder_path}"):
                if not overwrite:
                    print(f'{self.test_folder_path} already exits.')
                    raise Exception(
                        'Test folder already exists. If you want to overwrite, use flag -ow')
                else:
                    shutil.rmtree(self.test_folder_path)

        self.description = train_test_config['description']
        self.experiment_path = experiment_path
        self.experiment_name = self.experiment_path.split('/')[-1]
        if self.train:
            self.main_log_filename = f"{self.experiment_path}/logs/main.json"
            self.log_episode_filename = f"{self.experiment_path}/logs/$EPISODE.json"
            self.metrics_step_filename = f"{self.experiment_path}/logs/training/$STEP.json"
            self.reward_image_filename = f"{self.experiment_path}/plots/reward.png"
            self.reward_image_data_filename = f"{self.experiment_path}/plots/data.csv"
        else:
            self.main_log_filename = f"{self.test_folder_path}/logs/main.json"
            self.log_episode_filename = f"{self.test_folder_path}/logs/$EPISODE.json"
            self.reward_image_filename = f"{self.test_folder_path}/plots/reward.png"
            self.reward_image_data_filename = f"{self.test_folder_path}/plots/data.csv"

        self.fps = fps
        self.create_folders()
        
        self.main_logfile = self.init_main_log_file()
        
        self.reset_episodes_logfile()
        
        self.save_config_files(
            observation_config=observation_config, agent_config=agent_config, train_test_config=train_test_config)
        
        self.update_end_date()
        self.save_logfile(logfile=self.main_logfile, filename=self.main_log_filename)

        self.reset_monitor_buffer()
        
        if resume_training:
            create_empty_file(filename=f"{self.experiment_path}/resumed.txt")
        
        self.n_logfile_full = 0 if self.train else 0
        self.log_every = log_every
        
    
    def create_folders(self):
        if self.train:
            os.makedirs(self.experiment_path)
            os.makedirs(f'{self.experiment_path}/plots')
            os.makedirs(f'{self.experiment_path}/logs')
            os.makedirs(f'{self.experiment_path}/logs/training')
            os.makedirs(f'{self.experiment_path}/logs/training/plots')
            os.makedirs(f'{self.experiment_path}/weights')
            os.makedirs(f'{self.experiment_path}/monitor')
            os.makedirs(f'{self.experiment_path}/configs')
            os.makedirs(f'{self.experiment_path}/episode_stats')
            os.makedirs(f'{self.experiment_path}/weights/optimizers')
        else:
            os.makedirs(f'{self.test_folder_path}')
            os.makedirs(f'{self.test_folder_path}/monitor')
            os.makedirs(f'{self.test_folder_path}/episode_stats')
            os.makedirs(f'{self.test_folder_path}/plots')
            os.makedirs(f'{self.test_folder_path}/logs')

    def init_main_log_file(self):
        metadata = {'experiment_name': self.experiment_name,
                    'description': self.description,
                    'start_date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    'end_date': None,
                    'user': os.getenv('USER'),
                    'version': '0.0'
                    }

        return {'_metadata': metadata,
                'episode_saved': 0,
                'steps_saved': 0,
                'total_episodes': 0,
                'total_steps': 0}
    
    def reset_episodes_logfile(self):
        self.episodes_logfile = {}
    
    def init_episode_log_file(self):
        return {'steps' : {},
                'stats' : {}}
        
                    
    def reset_monitor_buffer(self):
        self.monitor_buffer = {}
        
    def reset(self):
        self.reset_episodes_logfile()
        self.reset_monitor_buffer()

    def save_config_files(self, observation_config, agent_config, train_test_config):

        env_config_name = train_test_config['env_name']
        # in IL_RL folder.
        env_config_path = f"{os.getenv('IL_RL_ROOT')}/config/envs/{env_config_name}"
        with open(env_config_path) as f:
            env_config = yaml.load(f, Loader=SafeLoader)

        observation_config_filename = f'{self.experiment_path}/configs/observation.yaml'
        agent_config_filename = f'{self.experiment_path}/configs/agent.yaml'
        train_test_config_filename = f'{self.experiment_path}/configs/{self.train_test_config_name}'
        env_config_filename = f'{self.experiment_path}/configs/{env_config_name}'

        if self.train:
            save_config(observation_config, observation_config_filename)
            save_config(agent_config, agent_config_filename)
            save_config(train_test_config, train_test_config_filename)
            save_config(env_config, env_config_filename)
        else:
            save_config(train_test_config, train_test_config_filename)
            save_config(env_config, env_config_filename)

    def add_data(self, episode, step, info, observation, action, reward, done):
        
        if episode not in self.episodes_logfile.keys():
            self.episodes_logfile[episode] = self.init_episode_log_file()
        if episode not in self.monitor_buffer.keys():
            self.monitor_buffer[episode] = []
        
        
        filtered_obs = self.filter_observation(observation=observation)
        
        self.episodes_logfile[episode]['steps'][step] = recursive_format({'info': info,
                                                                        'observation': filtered_obs,
                                                                        'action': action,
                                                                        'reward': reward}, format_number)
        monitor_image = observation['monitor']['data']
        self.monitor_buffer[episode].append(monitor_image)
        
        if done:
            self.episodes_logfile[episode]['stats'] = recursive_format({'episode_event': info['episode_event'],
                                                                'episode_stat': info['episode_stat']}, format_number)
                
      
            
    def save_episodes(self, episode, step, scores, steps_array, eval_idx):
        self.save_episodes_log_file(eval_idx)
        self.save_episode_video(eval_idx=eval_idx)
        self.save_reward_image(steps_array=steps_array,
                             scores=scores, moving_window=0)
        
        self.main_logfile['total_episodes'] = int(episode)
        self.main_logfile['total_steps'] = int(step)
        self.update_end_date()
        self.save_logfile(logfile=self.main_logfile, filename=self.main_log_filename)
               
        self.reset_episodes_logfile()
        self.reset_monitor_buffer()

    def save_episode_video(self, eval_idx):
        if self.train:
            for episode, images in self.monitor_buffer.items():
                video_name = f"{self.experiment_path}/monitor/{eval_idx}_{episode:04d}.mp4"
                save_video_from_images(video_name, images=images, fps=self.fps)
        else:
            for episode, images in self.monitor_buffer.items():
                video_name = f"{self.test_folder_path}/monitor/{eval_idx}_{episode:04d}.mp4"
                save_video_from_images(video_name, images=images, fps=self.fps)

        
    def save_episodes_log_file(self, eval_idx):
        for episode, episode_logfile in self.episodes_logfile.items():
            self.save_logfile(logfile=episode_logfile, filename= self.log_episode_filename.replace('$EPISODE', f'{eval_idx}_{episode:04d}'))
    
    def update_end_date(self):
        self.main_logfile['_metadata']['end_date'] = datetime.now().strftime(
                                        "%d/%m/%Y %H:%M:%S")

    def save_logfile(self, logfile, filename):
        with open(filename, "w") as f:
            json.dump(logfile, f, indent=4)
    
    def filter_observation(self, observation):
        speed = observation['speed']
        gnss = observation['gnss']
        waypoints = observation['waypoints']
        control = observation['control']

        filtered_obs = {'speed': convert_numpy_dict_to_list_dict(speed),
                        'gnss': convert_numpy_dict_to_list_dict(gnss),
                        'waypoints': convert_numpy_dict_to_list_dict(waypoints),
                        'control': convert_numpy_dict_to_list_dict(control)}
        return filtered_obs

    def update_episodes_saved(self, episode, step):
        self.main_logfile['episode_saved'] = int(episode)
        self.main_logfile['steps_saved'] = int(step)
        self.update_end_date()
        self.save_logfile(logfile=self.main_logfile, filename=self.main_log_filename)

    def save_reward_image(self, steps_array, scores, moving_window):
        
        save_episode_plot(x_array=steps_array, y_array=scores, moving_window=moving_window,
                          xlabel='Steps', ylabel='Reward', vertical_line=self.main_logfile['steps_saved'], filename=self.reward_image_filename)
        
        # save steps and scores to csv.
        data = zip(steps_array, scores)
        with open(self.reward_image_data_filename, "w") as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
        
    def is_full(self):
        if len(list(self.monitor_buffer.keys())) > self.n_logfile_full:
            return True
        return False

    def save_metrics(self, metrics, steps):
        if steps % self.log_every == 0:
            self.save_logfile(logfile=metrics, filename= self.metrics_step_filename.replace('$STEP', f'{steps:07d}'))
            
            if steps % (self.log_every * 10) == 0:
                self.save_plot_metrics()

    def save_plot_metrics(self):
        logs_dir = f"{self.experiment_path}/logs/training"
        files = sorted(filter(lambda x: os.path.isfile(os.path.join(logs_dir, x)),
                    os.listdir(logs_dir)))

        data = {'steps':[]}
        for file in files:
            metrics = get_config(f"{logs_dir}/{file}")
            for key, value in metrics.items():
                if key in data:
                    data[key].append(value)
                else:
                    data[key] = [value]
            if len(metrics.keys()) > 0: 
                data['steps'].append(int(file.split('.')[0]))
        
        for key, value in data.items():
            if key == 'steps':
                continue 
            save_episode_plot(x_array=data['steps'], y_array=value, moving_window=0, xlabel='Steps', ylabel=key, vertical_line=None, filename=f"{logs_dir}/plots/{key}.png")
            