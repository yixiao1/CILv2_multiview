import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import torch.optim as optimizer
import importlib
import sys
import carla

from colorama import Fore

from train_rl.agents.models.rl_networks.critic_network import CriticNetwork
from train_rl.agents.models.memory.memory_drqv2 import ReplayBufferStorage, make_replay_loader
from train_rl.utilities.networks import update_target_network, StdSchedule, RandomShiftsAug
from train_rl.utilities.configs import get_config, DotDict
from train_rl.utilities.distributions import TruncatedNormal

from dataloaders.transforms import encode_directions_4

class Agent():
    def __init__(self, training_config, critic_config, memory_config, exploration_config, experiment_path, init_memory, il_agent_config):
        
        self.experiment_path = experiment_path
        self.il_agent_config = il_agent_config
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
        self.grad_clip = training_config['grad_clip']

        experiment_name = experiment_path.split('/')[-1]

        if init_memory:
            replay_dir = f"{os.getenv('HOME')}/memory/{experiment_name}"
            self.replay_storage = ReplayBufferStorage(
                obs_info=self.obs_info, replay_dir=replay_dir)
            
            self.replay_loader = make_replay_loader(replay_dir=replay_dir, obs_info=self.obs_info, max_size=memory_config['capacity'], batch_size=self.batch_size, num_workers=memory_config['num_workers'], nstep=self.n_step, discount=self.discount_factor)
        
        # critic
        self.critic = CriticNetwork(state_size=self.state_size)
        self.critic_target = CriticNetwork(state_size=self.state_size)

        # hard update using tau=1.
        update_target_network(self.critic_target, self.critic, tau=1)

        self.il_agent = self.setup_il_agent()
        
        # get g_config which is like the policy config for the CILv2 model.
        parts = il_agent_config['agent-config'].rsplit('/', 1)
        filename = parts[0].split('/')[-1] + ".yaml"
        g_config_path = parts[0] + '/' + filename
        self.g_conf = DotDict(get_config(g_config_path))
        
        # create optimizers for critic and actor
        self.optimizer_critic = optimizer.Adam(self.critic.parameters(), lr=training_config['lr'])
        self.optimizer_actor = optimizer.Adam(self.il_agent._model.parameters(), lr=training_config['lr'])
        
        # create code to save the actor (using the same way of CILv2), and the critic, as an additional file.
        name = f"/checkpoints/CILv2_multiview_{experiment_name}.pth"
        self.cil_il_agent_model_path = il_agent_config['agent-config'].rsplit('/', 1)[0] + name
        
        self.il_agent_model_path = f"{experiment_path}/weights/CILv2_multiview.pth"
        self.il_agent_optim_path = f"{experiment_path}/weights/optimizers/CILv2_multiview_optimizer.pth"
        
        self.critic_path = f"{experiment_path}/weights/critic.pth"
        self.critic_optimizer_path = f"{experiment_path}/weights/critic_optimizer.pth"
        
        self.critic_target_path = f"{experiment_path}/weights/critic_target.pth"
        
        # init vars.
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
        module_name = os.path.basename(self.il_agent_config['agent']).split('.')[0]
        sys.path.insert(0, os.path.dirname(self.il_agent_config['agent']))
        module_agent = importlib.import_module(module_name)
        agent_class_name = getattr(module_agent, 'get_entry_point')()
        il_agent = getattr(module_agent, agent_class_name) \
            (self.il_agent_config['agent-config'], save_driving_vision=False,
                save_driving_measurement=False)

        il_agent._model.train()
        il_agent._model.to(self.device)
        
        for param in il_agent._model.parameters():
            param.requires_grad = False
            
        for param in il_agent._model._model.action_output.parameters():
                param.requires_grad = True     
        
        return il_agent       

    def encode(self, obs):
        
        # HAD TO CHANGE THE PROCESS_COMMAND!
        
        # use forward from the CILv2, with the exception of the action.
        # detach the latent from the graph.
        self.norm_rgb = [[self.il_agent.process_image(obs[camera_type]).unsqueeze(0).to(self.device) for camera_type in self.g_conf.DATA_USED]]
        

        self.norm_speed = [torch.FloatTensor([self.il_agent.process_speed(obs['speed'])]).to(self.device)]

       
       
        if self.g_conf.DATA_COMMAND_ONE_HOT:
            cmd = obs['command']
            self.direction = [torch.FloatTensor(encode_directions_4(cmd)).to(self.device).unsqueeze(0)]
        else:
            raise NotImplementedError
    

        state = self.IL_get_latent(self.norm_rgb, self.direction, self.norm_speed)
        
        state = state.detach()
        
        return state

    @torch.no_grad()
    def IL_get_latent(self, src_images, src_directions, src_speeds):
        
        S = int(self.g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = src_directions[0].shape[0]

        x = torch.stack([torch.stack(src_images[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = x.view(B * S * len(self.g_conf.DATA_USED), self.g_conf.IMAGE_SHAPE[0], self.g_conf.IMAGE_SHAPE[1], self.g_conf.IMAGE_SHAPE[2])  # [B*S*cam, 3, H, W]
        d = src_directions[-1]  # [B, 4]
        s = src_speeds[-1]  # [B, 1]
        
        # image embedding
        e_p, resnet_inter = self.il_agent._model._model.encoder_embedding_perception(x)  # [B*S*cam, dim, h, w]
        encoded_obs = e_p.view(B, S * len(self.g_conf.DATA_USED), self.il_agent._model._model.res_out_dim,  self.il_agent._model._model.res_out_h * self.il_agent._model._model.res_out_w)  # [B, S*cam, dim, h*w]
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.il_agent._model._model.res_out_dim)  # [B, S*cam*h*w, 512]
        e_d = self.il_agent._model._model.command(d).unsqueeze(1)  # [B, 1, 512]
        e_s = self.il_agent._model._model.speed(s).unsqueeze(1)  # [B, 1, 512]

        encoded_obs = encoded_obs + e_d + e_s   # [B, S*cam*h*w, 512]

        if self.il_agent._model._model.params['TxEncoder']['learnable_pe']:
            # positional encoding
            pe = encoded_obs + self.il_agent._model._model.positional_encoding    # [B, S*cam*h*w, 512]
        else:
            pe = self.il_agent._model._model.positional_encoding(encoded_obs)
            
        # Transformer encoder multi-head self-attention layers
        in_memory, attn_weights = self.il_agent._model._model.tx_encoder(pe)  # [B, S*cam*h*w, 512]
        in_memory = torch.mean(in_memory, dim=1)  # [B, 512]

        return in_memory

    @torch.no_grad()
    def choose_action(self, obs, step, training=True):

        obs = self.filter_obs(obs=obs)

        state = self.encode(obs=obs)
                
        std = self.std_schedule.get(step=step)

        if training:
            action = self.sample_action(state, std, clip=None)
            
        else:
            action = self.determinitistic_action(state)
            
        action = action.detach().cpu().numpy().squeeze()

        steer, throttle, brake = self.il_agent.process_control_outputs(action)
        controls = carla.VehicleControl(throttle= float(throttle), steer=float(steer), brake=float(brake))

        return action, controls

    def sample_action(self, state, std, clip=None):
        mean = self.determinitistic_action(state)
        dist = TruncatedNormal(mean, std)
        action = dist.sample(clip=clip)
        return action
        
    def determinitistic_action(self, state):
        return self.il_agent._model._model.action_output(state).unsqueeze(1)
    
    def random_action(self, obs):

        action = np.asarray([random.uniform(-1, 1), random.uniform(-1, 1)])
        steer, throttle, brake = self.il_agent.process_control_outputs(action)
        controls = carla.VehicleControl(throttle= float(throttle), steer=float(steer), brake=float(brake))
        
        return action, controls

    def filter_obs(self, obs):
        obs_ = {}
        
        obs_['rgb_left'] = obs['rgb_left']['data']
        obs_['rgb_central'] = obs['rgb_central']['data']
        obs_['rgb_right'] = obs['rgb_right']['data']
        obs_['speed'] = obs['SPEED']['forward_speed']
        obs_['GPS'] = obs['GPS']['gnss']
        obs_['command'] = obs['GPS']['command']
        obs_['IMU'] = obs['GPS']['imu']

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
        next_obs = {'state' : self.encode(next_obs).detach().cpu().numpy()}
        
        self.replay_storage.add(
            action=action, reward=reward, next_obs=next_obs, done=done)

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
        state_batch, action_batch, reward_batch, discount_batch, next_state_batch, done_batch = tuple(
            next(self.replay_iter))

        state_batch = self.convert_obs_into_device(
            state_batch)
        next_state_batch = self.convert_obs_into_device(
            next_state_batch)
        
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        discount_batch = discount_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # critic networks.
        metrics.update(self.update_critics(
                state_batch=state_batch, action_batch=action_batch, reward_batch=reward_batch, discount_batch=discount_batch,
                next_state_batch=next_state_batch, done_batch=done_batch, step=step))
        
        # actor networks.
        metrics.update(self.update_policy(state_batch=state_batch, step=step))
        
        # metrics update.
        if self.train_ctn % self.target_update_interval == 0:
            update_target_network(target=self.critic_target,
                                  source=self.critic, tau=self.critic_tau)
            
        return metrics

    def update_critics(self, state_batch, action_batch, reward_batch, discount_batch, next_state_batch, done_batch, step):
        metrics = dict()
        
        # get Q_target using Q(s) = r + gamma*Q(s').
        with torch.no_grad():
            std = self.std_schedule.get(step)
            next_state_action = self.sample_action(
                next_state_batch, std=std, clip=self.std_clip)
            q1_next_target, q2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_q_next_target = torch.min(
                q1_next_target, q2_next_target)
            q_value_target = reward_batch + (discount_batch * min_q_next_target)

        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, q_value_target)
        q2_loss = F.mse_loss(q2, q_value_target)
        q_loss = q1_loss + q2_loss

        self.optimizer_critic.zero_grad(set_to_none=True)

        q_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

        self.optimizer_critic.step()
        
        metrics['critic_loss'] = round(q_loss.item(), 4)

        return metrics

    def update_policy(self, state_batch, step):
        metrics = dict()

        std = self.std_schedule.get(step)

        actions = self.sample_action(state_batch, std=std, clip=self.std_clip)

        q1, q2 = self.critic(state_batch, actions)

        min_q = torch.min(q1, q2)

        policy_loss = -min_q.mean()

        self.optimizer_actor.zero_grad(set_to_none=True)
        
        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.il_agent._model.parameters(), self.grad_clip)

        self.optimizer_actor.step()
        
        metrics['policy_loss'] = round(policy_loss.item(), 4)

        return metrics
    
    def reset(self, obs):
        pass

    def set_train_mode(self):
        self.il_agent._model.train()
        self.critic.train()
        self.critic_target.train()

    def set_eval_mode(self):
        self.il_agent._model.eval()
        self.critic.eval()
        self.critic_target.eval()

    def save_models(self, save_memory=False):
        print(f'{Fore.GREEN} saving models... {Fore.RESET}')

        il_model_state_dict = self.il_agent._model.state_dict()
        il_model_optimizer_state_dict = self.optimizer_actor.state_dict()
        
        torch.save(il_model_state_dict, self.cil_il_agent_model_path)
        torch.save(il_model_state_dict, self.il_agent_model_path)

        torch.save(il_model_optimizer_state_dict, self.il_agent_model_path)
        
        torch.save(self.critic.state_dict(), self.critic_path)
        torch.save(self.optimizer_critic.state_dict(), self.critic_optimizer_path)
        
        torch.save(self.critic_target.state_dict(), self.critic_target_path)
        
    
    def load_models(self, save_memory=False):
        print(f'{Fore.GREEN} loading models... {Fore.RESET}')
        
        # load everything from my files!
        
        self.il_agent._model.load_state_dict(torch.load(self.il_agent_model_path))
        self.optimizer_actor.load_state_dict(torch.load(self.il_agent_optim_path))
        
        self.critic.load_state_dict(torch.load(self.critic_path))
        self.optimizer_critic.load_state_dict(torch.load(self.critic_optimizer_path))
        self.critic_target.load_state_dict(torch.load(self.critic_target_path))

