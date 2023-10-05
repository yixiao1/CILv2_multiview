import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os

from colorama import Fore
from importlib import import_module

from train_rl.agents.models.memory.memory_drqv2 import ReplayBufferStorage, make_replay_loader
from utilities.controls import carla_control, PID
from utilities.networks import update_target_network, StdSchedule, RandomShiftsAug


class Agent():
    def __init__(self, training_config, critic_config, memory_config, exploration_config, experiment_path, init_memory, il_agent):
        
        self.experiment_path = experiment_path
        self.device = torch.device(training_config['device'])
        self.batch_size = training_config['batch_size']
        self.discount_factor = training_config['discount_factor']
        self.state_size = 512
        self.target_update_interval = training_config['target_update_interval']
        self.obs_info = self.parse_obs_info(
            memory_config['obs_info'])
        self.repeat_action = training_config['repeat_action']
        self.n_step = training_config['n_step']
        self.std_clip = exploration_config['std_clip']
        self.std_schedule = StdSchedule(
            init=exploration_config['std_schedule_init'], final=exploration_config['std_schedule_final'], duration=exploration_config['std_schedule_duration'])
        self.critic_tau = critic_config['tau']
        self.il_agent = il_agent

        if init_memory:
            experiment_name = experiment_path.split('/')[-1]
            replay_dir = f"{os.getenv('HOME')}/memory/{experiment_name}"
            self.replay_storage = ReplayBufferStorage(
                obs_info=self.obs_info, replay_dir=replay_dir)
            
            self.replay_loader = make_replay_loader(replay_dir=replay_dir, obs_info=self.obs_info, max_size=memory_config['capacity'], batch_size=self.batch_size, num_workers=memory_config['num_workers'], nstep=self.n_step, discount=self.discount_factor)
        
        # critic
        module_str, class_str = critic_config['entry_point'].split(':')
        _Class = getattr(import_module(module_str), class_str)
        self.critic = _Class(state_size=self.state_size)
        self.critic_target = _Class(state_size=self.state_size)

        # hard update using tau=1.
        update_target_network(self.critic_target, self.critic, tau=1)

        self.setup_il_agent()
        
        # create optimizers for critic and actor
        
        # create code to save the actor (using the same way of CILv2), and the critic, as an additional file.
        
        
        # init vars.
        self.action_ctn = 0
        self.prev_action = None
        self.train_ctn = 0
        self._replay_iter = None
    
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @staticmethod
    def parse_obs_info(obs_info):
        for state_key, state_value in obs_info.items():
            for key, value in state_value.items():
                obs_info[state_key][key] = eval(value)
        return obs_info

    def setup_il_agent(self):
        # set model to train
        # freeze all layers except action
        
    

    def encode(self, obs, detach=False):
        
        # use forward from the CILv2, with the exception of the action.
        
        
        image = obs['image']  # (N, 3, 232, 232).
        waypoints = obs['waypoints']  
        vm = obs['vehicle_measurements']

        state_image = self.image_encoder(image)
        state_waypoints = self.waypoint_encoder(waypoints)
        state_vm = self.vm_encoder(vm)
            
        if detach:
            state_image = state_image.detach()
            state_waypoints = state_waypoints.detach()
            state_vm = state_vm.detach()

        state = torch.cat([state_image, state_waypoints, state_vm], dim=1)  # (N, 519)

        return state

    @torch.no_grad()
    def choose_action(self, obs, step, greedy=False):

        current_velocity = self.get_current_speed(obs=obs)

        if self.action_ctn % self.repeat_action == 0:
            obs = self.filter_obs(obs=obs)
            obs = self.convert_obs_into_torch(
                obs=obs, unsqueeze=True)

            state = self.encode(obs=obs, detach=True)
            std = self.std_schedule.get(step=step)

            if greedy is False:
                action, _ = self.policy.sample(state, std, clip=None)
            else:
                _, action = self.policy.sample(state, std, clip=None)

            action = action.detach().cpu().numpy()[0]

        else:
            action = self.prev_action

        controls = carla_control(self.pid.get(
            action=action, velocity=current_velocity))

        self.action_ctn += 1
        self.prev_action = action

        return action, controls

    def random_action(self, obs):

        current_velocity = self.get_current_speed(obs=obs)

        if self.action_ctn % self.repeat_action == 0:

            action = np.asarray([random.uniform(0, 1), random.uniform(-1, 1)])
        else:
            action = self.prev_action

        controls = carla_control(self.pid.get(
            action=action, velocity=current_velocity))

        self.action_ctn += 1
        self.prev_action = action

        return action, controls

    def filter_obs(self, obs):
        obs_ = {}

        resized_image = cv2.resize(
            obs['image']['data'], self.obs_info['image']['shape'][1:3], interpolation=cv2.INTER_AREA)

        obs_['image'] = np.einsum(
            'kij->jki', resized_image)

        obs_['waypoints'] = np.array(
            obs['waypoints']['location'])[0:self.num_waypoints, 0:2].reshape(self.num_waypoints, 2)

        obs_speed = np.array(
            obs['speed']['speed'][0] / self.maximum_speed, dtype=np.float32).reshape(1)

        obs_steer = np.array(
            obs['control']['steer'][0]).reshape(1)

        obs_['vehicle_measurements'] = np.concatenate([obs_speed, obs_steer]).reshape(2)
        
        return obs_

    def convert_obs_into_torch(self, obs, unsqueeze=False):
        for key, value in obs.items():
            if unsqueeze:
                obs[key] = torch.from_numpy(
                    value).to(self.device).unsqueeze(0)
            else:
                obs[key] = torch.from_numpy(value).to(self.device)
        return obs

    def convert_obs_into_device(self, obs, unsqueeze=False):
        for key, value in obs.items():
            if unsqueeze:
                obs[key] = value.to(self.device).unsqueeze(0)
            else:
                obs[key] = value.to(self.device)
        return obs

    def remember(self, obs, action, reward, next_obs, done):
        obs = None
        next_obs = self.filter_obs(next_obs)
        self.replay_storage.add(
            action=action, reward=reward, next_obs=next_obs, done=done)

    def augment_obs(self, obs):
        obs['image'] = self.aug(obs['image'].float())
        return obs

    def clone_obs(self, obs):
        obs_ = {}
        for key, value in obs.items():
            obs_[key] = value.clone()

        return obs_

    def train(self, step):
        self.train_ctn += 1

        metrics = dict()
        
        if self.train_ctn < 512:
            return metrics

        # sample batch from memory.
        obs_batch, action_batch, reward_batch, discount_batch, next_obs_batch, done_batch = tuple(
            next(self.replay_iter))

        obs_batch = self.convert_obs_into_device(
            obs_batch)
        next_obs_batch = self.convert_obs_into_device(
            next_obs_batch)
        
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        discount_batch = discount_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # critic networks.
        metrics.update(self.update_critics(
                obs_batch=obs_batch, action_batch=action_batch, reward_batch=reward_batch, discount_batch=discount_batch,
                next_obs_batch=next_obs_batch, done_batch=done_batch, step=step))
        
        # actor networks.
        metrics.update(self.update_policy(obs_batch=obs_batch, step=step))
        
        # metrics update.
        if self.train_ctn % self.target_update_interval == 0:
            update_target_network(target=self.critic_target,
                                  source=self.critic, tau=self.critic_tau)
            
        return metrics

    def update_critics(self, obs_batch, action_batch, reward_batch, discount_batch, next_obs_batch, done_batch, step):
        metrics = dict()
        
        # get Q_target using Q(s) = r + gamma*Q(s').
        with torch.no_grad():
            std = self.std_schedule.get(step)
            next_state_batch = self.encode(next_obs_batch)
            next_state_action, _ = self.policy.sample(
                next_state_batch, std=std, clip=self.std_clip)
            q1_next_target, q2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_q_next_target = torch.min(
                q1_next_target, q2_next_target)
            q_value_target = reward_batch + (discount_batch * min_q_next_target)

        state_batch = self.encode(obs_batch)
        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, q_value_target)
        q2_loss = F.mse_loss(q2, q_value_target)
        q_loss = q1_loss + q2_loss

        self.vm_encoder.optimizer.zero_grad(set_to_none=True)
        self.waypoint_encoder.optimizer.zero_grad(set_to_none=True)
        self.image_encoder.optimizer.zero_grad(set_to_none=True)
        self.critic.optimizer.zero_grad(set_to_none=True)

        q_loss.backward()

        self.vm_encoder.optimizer.step()
        self.waypoint_encoder.optimizer.step()
        self.image_encoder.optimizer.step()
        self.critic.optimizer.step()
        
        metrics['critic_loss'] = round(q_loss.item(), 4)

        return metrics

    def update_policy(self, obs_batch, step):
        metrics = dict()

        std = self.std_schedule.get(step)

        state_batch = self.encode(obs_batch, detach=True)
        actions, _ = self.policy.sample(state_batch, std=std, clip=self.std_clip)

        q1, q2 = self.critic(state_batch, actions)

        min_q = torch.min(q1, q2)

        policy_loss = -min_q.mean()

        self.policy.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy.optimizer.step()
        
        metrics['policy_loss'] = round(policy_loss.item(), 4)

        return metrics
    
    @staticmethod
    def get_current_speed(obs):
        return obs['speed']['speed'][0]

    def reset(self, obs):
        self.pid.reset()

    def set_train_mode(self):
        self.critic.train()
        self.critic_target.train()
        self.policy.train()
        self.image_encoder.train()
        self.waypoint_encoder.train()
        self.vm_encoder.train()

    def set_eval_mode(self):
        self.critic.eval()
        self.critic_target.eval()
        self.policy.eval()
        self.image_encoder.eval()
        self.waypoint_encoder.eval()
        self.vm_encoder.eval()

    def save_models(self, save_memory=False):
        print(f'{Fore.GREEN} saving models... {Fore.RESET}')

        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.policy.save_checkpoint()
        self.image_encoder.save_checkpoint()
        self.waypoint_encoder.save_checkpoint()
        self.vm_encoder.save_checkpoint()


    
    def load_models(self, save_memory=False):
        print(f'{Fore.GREEN} loading models... {Fore.RESET}')

        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
        self.policy.load_checkpoint()
        self.image_encoder.load_checkpoint()
        self.waypoint_encoder.load_checkpoint()
        self.vm_encoder.load_checkpoint()

