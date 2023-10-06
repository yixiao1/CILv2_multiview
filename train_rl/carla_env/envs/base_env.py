#!/usr/bin/env python3

"""Adapted from https://github.com/zhejz/carla-roach CC-BY-NC 4.0 license."""

import carla
import numpy as np


from train_rl.carla_env.core.obs_manager.obs_manager_handler import ObsManagerHandler
from train_rl.carla_env.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from train_rl.carla_env.core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from train_rl.carla_env.core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from train_rl.carla_env.core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from train_rl.carla_env.utils.dynamic_weather import WeatherHandler
from train_rl.carla_env.utils.traffic_light import TrafficLightHandler
from train_rl.utilities.common import set_random_seed


class CarlaEnv():
    def __init__(self, carla_map_train, carla_map_val, host, port, obs_configs, terminal_configs, reward_configs, all_tasks_train, all_tasks_val, carla_fps, tm_port, seed):
        self._all_tasks_train = all_tasks_train
        self._all_tasks_val = all_tasks_val
        self._obs_configs = obs_configs
        self._carla_map_train = carla_map_train
        self._carla_map_val = carla_map_val
        self._reward_configs = reward_configs
        self._terminal_configs = terminal_configs
        self.carla_fps = carla_fps
        self.port = port 
        self.host = host
        self.val = False


        self.seed = seed

        self.name = self.__class__.__name__
 
        self._tm_port = tm_port

        self._init_client(carla_map=carla_map_train, host=host, port=port)

        # define observation space handler.
        self._om_handler = ObsManagerHandler(obs_configs=obs_configs)

        # define ego vehicle handler.
        self._ev_handler = EgoVehicleHandler(
            self._client, reward_configs, terminal_configs)

        # define the zombie walkers handler.
        self._zw_handler = ZombieWalkerHandler(self._client)

        # define the zombie vehicles handler.
        self._zv_handler = ZombieVehicleHandler(
            self._client, self._tm.get_port())

        # define the scenario actor handler.
        self._sa_handler = ScenarioActorHandler(self._client)

        # define the weather
        self._w_handler = WeatherHandler(self._world)

        self._task_idx = 0
        self._shuffle_task = True
        self._task = self._all_tasks_train[self._task_idx].copy()
        
        
    def set_task_idx(self, task_idx):
        self._task_idx = task_idx
        self._shuffle_task = False
        if self.val:
            self._task = self._all_tasks_val[self._task_idx].copy()
        else:
            self._task = self._all_tasks_train[self._task_idx].copy()

    @property
    def num_tasks(self):
        if self.val:
            return len(self._all_tasks_val)
        else:
            return len(self._all_tasks_train)
    
    @property
    def task(self):
        return self._task

    def _init_client(self, carla_map, host, port):
        client = None
        while client is None:
            try:
                client = carla.Client(host, port)
                client.set_timeout(10.0)
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                client = None

        self._client = client

        self._world = self._client.load_world(carla_map)
        self._tm = self._client.get_trafficmanager(self._tm_port)

        self._set_sync_mode(True)
        
        self._tm.set_random_device_seed(self.seed)

        self._world.tick()

        TrafficLightHandler.reset(self._world)
        
        
    def change_world(self, val):
        self.clean()
        if val:
            self._world = self._client.load_world(self._carla_map_val)
            self.val = True
        else:
            self._world = self._client.load_world(self._carla_map_train)
            self.val = False

        self._tm = self._client.get_trafficmanager(self._tm_port)
        self._set_sync_mode(True)
        self._tm.set_random_device_seed(self.seed)
        self._world.tick()
        TrafficLightHandler.reset(self._world)


        self._om_handler = ObsManagerHandler(obs_configs=self._obs_configs)
        self._ev_handler = EgoVehicleHandler(
            self._client, self._reward_configs, self._terminal_configs)
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._zv_handler = ZombieVehicleHandler(
            self._client, self._tm.get_port())
        self._sa_handler = ScenarioActorHandler(self._client)
        self._w_handler = WeatherHandler(self._world)
        

    def _set_sync_mode(self, sync):
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 1.0 / self.carla_fps
        settings.deterministic_ragdolls = True
        self._world.apply_settings(settings)
        # if we are in synchronous mode we must ensure the traffic manager is also synchronous.
        self._tm.set_synchronous_mode(sync)

    def reset(self):
        if self._shuffle_task:
            self._task_idx = np.random.choice(self.num_tasks)
            if self.val:
                self._task = self._all_tasks_val[self._task_idx].copy()
            else: 
                self._task = self._all_tasks_train[self._task_idx].copy()
                
        self.clean()

        self._w_handler.reset(self._task['weather'])
                
        ev_spawn_location = self._ev_handler.reset(
            self._task['ego_vehicles'])

        self._sa_handler.reset(
            self._task['scenario_actors'], self._ev_handler.ego_vehicle)

        self._zw_handler.reset(
            self._task['num_zombie_walkers'], ev_spawn_location)

        self._zv_handler.reset(
            self._task['num_zombie_vehicles'], ev_spawn_location)

        self._om_handler.reset(self._ev_handler.ego_vehicle)

        self._world.tick()

        snapshot = self._world.get_snapshot()
        self._timestamp = {
            'step': 0,
            'frame': snapshot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snapshot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snapshot.timestamp.elapsed_seconds,
            'start_frame': snapshot.timestamp.frame,
            'start_wall_time': snapshot.timestamp.platform_timestamp,
            'start_simulation_time': snapshot.timestamp.elapsed_seconds
        }

        _, _, _ = self._ev_handler.tick(self.timestamp)
        # get obeservations
        obs_dict = self._om_handler.get_observation()


        obs_dict['is_first'] = True
        obs_dict['is_last'] = False
        
        return obs_dict

    def clean(self):
        self._om_handler.clean()
        self._ev_handler.clean()
        self._zw_handler.clean()
        self._zv_handler.clean()
        self._sa_handler.clean()
        self._w_handler.clean()
        self._world.tick()

    def step(self, control):
        self._ev_handler.apply_control(control)
        self._sa_handler.tick()
        self._world.tick()

        # update timestamp
        snap_shot = self._world.get_snapshot()
        self._timestamp['step'] = snap_shot.timestamp.frame - \
            self._timestamp['start_frame']
        self._timestamp['frame'] = snap_shot.timestamp.frame
        self._timestamp['wall_time'] = snap_shot.timestamp.platform_timestamp
        self._timestamp['relative_wall_time'] = self._timestamp['wall_time'] - \
            self._timestamp['start_wall_time']
        self._timestamp['simulation_time'] = snap_shot.timestamp.elapsed_seconds
        self._timestamp['relative_simulation_time'] = self._timestamp['simulation_time'] \
            - self._timestamp['start_simulation_time']

        reward, done, info = self._ev_handler.tick(
            self.timestamp)

        # get observations
        obs_dict = self._om_handler.get_observation()

            
        # update weather
        self._w_handler.tick(snap_shot.timestamp.delta_seconds)
        
        
        obs_dict['is_first'] = False
        obs_dict['is_last'] = done

        return obs_dict, reward, done, info

    @property
    def timestamp(self):
        return self._timestamp.copy()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.clean()
        self._set_sync_mode(False)
        self._client = None
        self._world = None
        self._tm = None

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        