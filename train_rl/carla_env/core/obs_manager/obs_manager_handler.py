from importlib import import_module

class ObsManagerHandler(object):
    def __init__(self, obs_configs):
        self._obs_managers = {}
        self._obs_configs = obs_configs
        self._init_obs_managers()

    def get_observation(self):
        obs_dict = {}
        om_dict = self._obs_managers
        for obs_id, om in om_dict.items():
            obs_dict[obs_id] = om.get_observation()
        return obs_dict


    def reset(self, ego_vehicle):
        self._init_obs_managers()
        for obs_id, om in self._obs_managers.items():
            om.attach_ego_vehicle(ego_vehicle)

    def clean(self):
        for obs_id, om in self._obs_managers.items():
            om.clean()
        self._obs_managers = {}

    def _init_obs_managers(self):
        self._obs_managers = {}
        for obs_id, obs_config in self._obs_configs.items():
            ObsManager = getattr(import_module('train_rl.carla_env.core.obs_manager.'+obs_config["module"]), 'ObsManager')
            self._obs_managers[obs_id] = ObsManager(obs_config)