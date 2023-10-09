import pygame
import time
import argparse
import sys
import os

from train_rl.visualization.src.monitor import Monitor
from train_rl.utilities.configs import get_env_config_from_train_test_config, get_config

def main():
   
    parser = argparse.ArgumentParser(description='Monitor')
    parser.add_argument('-en', '--experiment_name', type=str, required=True)
    parser.add_argument('-ei', '--eval_idx', type=int, required=True)
    parser.add_argument('-ep', '--episode', type=int, required=True)
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-rv', '--rlad_video', action='store_true')
    parser.add_argument('-fps', '--fps', type=int, default=0)
    parser.add_argument('-ttc', '--train_test_config', type=str, default='train.yaml')
    
    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))
    
    experiment_name = args['experiment_name']
    experiment_path = f"{os.getenv('HOME')}/results/CILv2_multiview/{experiment_name}"
    episode = args['episode']
    eval_idx =args['eval_idx']
    save = args['save']
    fps = args['fps']
    train_test_config_name = args['train_test_config']
    
    # if fps==0, we want to use the fps from the config file. Preferred option.
    if fps == 0:
        fps = int(get_env_config_from_train_test_config(experiment_name, train_test_config_name)[0]['env_configs']['carla_fps'])
    
    # init pygame.
    pygame.init()
    pygame.font.init()
    
    obs_config = get_config(f'{experiment_path}/configs/observation.yaml')
    width = obs_config['rgb_backontop']['width']
    height = obs_config['rgb_backontop']['height']
    
    monitor = Monitor(experiment_name=experiment_name, episode=episode, eval_idx=eval_idx, fps=fps, dim=(width, height), save=save, train_test_config_name=train_test_config_name, rlad_video=args['rlad_video'])
    
    display = pygame.display.set_mode(
    (width, height),
    pygame.HWSURFACE | pygame.DOUBLEBUF)
    display.fill((0, 0, 0))
    pygame.display.flip()

    while True:
        done = monitor.tick()
        if done:
            break
        monitor.render(display=display)
        pygame.display.flip()
        time.sleep(1/fps)
        

        
        


if __name__ == "__main__":
    main()

