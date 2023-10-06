import os
import yaml
import cv2
import argparse
import sys
import numpy as np
import wandb


from yaml.loader import SafeLoader
from colorama import Fore
from importlib import import_module

from train_rl.carla_env.envs.suites.endless_env import EndlessEnv
from train_rl.carla_env.envs.suites.nocrash_env import NoCrashEnv
from train_rl.run.src.log_file import LogFile
from train_rl.run.src.evaluate_agent import evaluate_agent
from train_rl.utilities.common import set_random_seed
from train_rl.utilities.configs import get_config_with_env_var


wandb.login()

def main():

    parser = argparse.ArgumentParser(description='Train/Test RL agents')
    parser.add_argument('-en', '--experiment_name', type=str, required=True)
    parser.add_argument('-ow', '--overwrite', action='store_true')
    parser.add_argument('-vm', '--visualize_monitor', action='store_true')
    parser.add_argument('-ttc', '--train_test_config',
                        type=str, default='train.yaml')
    parser.add_argument('-rt', '--resume_training', action='store_true')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    experiment_name = args['experiment_name']
    experiment_path = f'{os.getenv("HOME")}/results/CILv2_multiview/{experiment_name}'
    train_test_config_name = args['train_test_config']

    train = True if train_test_config_name == 'train.yaml' else False
    if train:
        print(f"{Fore.WHITE} Initialize training! {Fore.RESET}")
    else:
        print(f"{Fore.WHITE} Initialize testing! {Fore.RESET}")

    TRAINING_ROOT = os.getenv('TRAINING_ROOT')
    observation_config_path = f"{TRAINING_ROOT}/train_rl/config/experiments/{experiment_name}/observation.yaml"
    agent_config_path = f"{TRAINING_ROOT}/train_rl/config/experiments/{experiment_name}/agent.yaml"
    il_agent_config_path = f"{TRAINING_ROOT}/train_rl/config/experiments/{experiment_name}/il_agent.yaml"
    train_test_config_path = f"{TRAINING_ROOT}/train_rl/config/experiments/{experiment_name}/{train_test_config_name}"

    with open(observation_config_path) as f:
        observation_config = yaml.load(f, Loader=SafeLoader)
    with open(agent_config_path) as f:
        agent_config = yaml.load(f, Loader=SafeLoader)
    with open(train_test_config_path) as f:
        train_test_config = yaml.load(f, Loader=SafeLoader)
    
    il_agent_config = get_config_with_env_var(il_agent_config_path)
    
 


    env_config_name = train_test_config['env_name']
    env_config_path = f"{TRAINING_ROOT}/train_rl/config/envs/{env_config_name}"
    with open(env_config_path) as f:
        env_config = yaml.load(f, Loader=SafeLoader)[0]

    # set random seed
    seed = train_test_config.get('seed', None)
    set_random_seed(seed=seed)

    # Load Environment.
    carla_map_train = env_config['env_configs'].get('carla_map_train', None)
    carla_map_val = env_config['env_configs'].get('carla_map_val', None)
    carla_fps = env_config['env_configs'].get('carla_fps', None)
    host = train_test_config.get('host', None)
    port = train_test_config.get('port', None)
    tm_port = train_test_config.get('tm_port', None)
    num_zombie_vehicles = env_config['env_configs'].get(
        'num_zombie_vehicles', None)
    num_zombie_walkers = env_config['env_configs'].get(
        'num_zombie_walkers', None)
    route_description = env_config['env_configs'].get(
        'route_description', None)
    weather_group_train = env_config['env_configs'].get('weather_group_train', None)
    weather_group_val = env_config['env_configs'].get('weather_group_val', None)
    terminal_configs = env_config['env_configs'].get('terminal', None)
    reward_configs = env_config['env_configs'].get('reward', None)
    background_traffic = env_config['env_configs'].get(
        'background_traffic', None)
    task_type = env_config['env_configs'].get('task_type', None)
    routes_group = env_config['env_configs'].get('routes_group', None)
        
    # init agent.
    agent_config['kwargs']['experiment_path'] = experiment_path
    maximum_speed = env_config['env_configs']['reward']['kwargs']['maximum_speed']
    agent_config['kwargs']['init_memory'] = train
    agent_config['kwargs']['il_agent_config'] = il_agent_config
    module_str, class_str = agent_config['entry_point'].split(':')
    _Class = getattr(import_module(module_str), class_str)
    agent = _Class(**agent_config.get('kwargs', {}))

    # load weights.
    if not train:
        agent.load_models()
        agent.set_eval_mode()

    if args['resume_training']:
        agent.load_models()

    # init logfile.
    logfile = LogFile(experiment_path=experiment_path, observation_config=observation_config,
                      agent_config=agent_config, train_test_config=train_test_config,
                      train_test_config_name=train_test_config_name, overwrite=args['overwrite'],
                      fps=carla_fps, resume_training=args['resume_training'], log_every=agent_config['kwargs']['training_config']['log_every'])

    if env_config['env_id'].split('-')[0] == 'NoCrash':
        env = NoCrashEnv(carla_map_train=carla_map_train, carla_map_val=carla_map_val, host=host, port=port, obs_configs=observation_config,
                         reward_configs=reward_configs, terminal_configs=terminal_configs,
                         weather_group_train=weather_group_train, weather_group_val=weather_group_val, route_description=route_description,
                         background_traffic=background_traffic, carla_fps=carla_fps, tm_port=tm_port, seed=seed)
    elif env_config['env_id'].split('-')[0] == 'Endless':
        env = EndlessEnv(carla_map_train=carla_map_train, carla_map_val=carla_map_val, host=host, port=port, obs_configs=observation_config,
                         terminal_configs=terminal_configs, reward_configs=reward_configs,
                         num_zombie_vehicles=num_zombie_vehicles, num_zombie_walkers=num_zombie_walkers,
                         weather_group_train=weather_group_train, weather_group_val=weather_group_val, carla_fps=carla_fps, tm_port=tm_port, seed=seed)
    else:
        env = None
        raise Exception('Unknown Environment!')

    if env_config['env_id'].split('-')[0] == 'Endless' and not train:
        raise Exception('Endless Environment cannot be use for test.')

    # Init Training/Testing.
    maximum_steps = train_test_config.get('n_steps', None)
    warmup_steps = train_test_config.get('warmup_steps', None)
    start_remembering = train_test_config.get('start_remembering')
    eval_freq = train_test_config.get('evaluate_frequency_steps', None)
    eval_average_episodes = train_test_config.get(
        'evaluate_average_episodes', None)

    # if n_episodes==-1 means we want to test all tasks of the environment.
    if maximum_steps == -1:
        n_episodes = len(env._all_tasks)
    else:
        n_episodes = int(10e10)

    steps = 0
    steps_eval = 0
    eval_idx = 0
    scores, steps_array = [], []
    best_score = -np.inf

    
    if train:
        wandb.init(project='CILv2_multiview', name=experiment_name, config=agent_config, dir=experiment_path)
    
    # agent.watch_models()
    

    for ep in range(n_episodes):

        # evaluate agent.
        if (not train) or (steps_eval > eval_freq) or (steps == 0):
            num_episodes = eval_average_episodes if train else n_episodes
            eval_idx, best_score, scores, steps_array = evaluate_agent(env=env, agent=agent, logfile=logfile, step_train=steps, episode_train=ep-1,
                                                                       num_episodes=num_episodes, eval_idx=eval_idx, best_score=best_score, steps_array=steps_array, scores_eval=scores, maximum_speed=maximum_speed, train=train)
            if not train:
                break
            steps_eval = 0

        remember_cnt = 0
        done = False
        observation = env.reset()
        agent.reset(obs=observation)

        score = 0

        while not done:

            if steps < warmup_steps:
                action, controls = agent.random_action(obs=observation)
            else:
                action, controls = agent.choose_action(
                    obs=observation, step=steps, training=True)

            observation_, reward, done, info = env.step(controls)
            score += reward

            if remember_cnt > start_remembering:
                agent.remember(obs=observation, action=action,
                               reward=reward, next_obs=observation_, done=done)
            else:
                remember_cnt += 1

            metrics = agent.train(step=steps)

            if args['visualize_monitor']:
                image = observation_['monitor']['data']
                cv2.imshow('monitor', image)
                cv2.waitKey(1)

            observation = observation_
            steps += 1
            steps_eval += 1
            
            logfile.save_metrics(metrics, steps)
            
          
                
            wandb.log(data=metrics, step=steps)

        print(
            f" {Fore.BLUE} episode: {ep}, steps: {steps}, score: {score:.1f} {Fore.RESET}")

        if steps >= maximum_steps:
            break


if __name__ == '__main__':
    main()
