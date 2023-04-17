import os
import random
import torch
import glob
from torch.nn import DataParallel
import numpy as np

from .utils import sort_nicely
from configs import g_conf


class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __len__(self):
        return len(self.module)


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def check_saved_checkpoints(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        return None
    else:
        checkpoints = glob.glob(os.path.join(checkpoints_path, '*.pth'))
        if checkpoints:
            sort_nicely(checkpoints)
            return checkpoints[-1]
        else:
            return None


def check_saved_checkpoints_in_total(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        return None
    else:
        checkpoints = glob.glob(os.path.join(checkpoints_path, '*.pth'))
        if checkpoints:
            sort_nicely(checkpoints)
            return checkpoints
        else:
            return None


def update_learning_rate(optimizer: torch.optim.Optimizer,
                         iteration: int = None,
                         total_iterations: int = None,
                         min_lr: float = g_conf.LEARNING_RATE_MINIMUM) -> None:
    """ Adjusts the learning rate based on the schedule """
    # For g_conf.LEARNING_RATE_SCHEDULE == 'step'
    if g_conf.LEARNING_RATE_POLICY['name'] == 'normal':
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
            print('Previous lr:', cur_lr)
            new_lr = cur_lr * g_conf.LEARNING_RATE_POLICY['level']
            param_group['lr'] = max(new_lr, min_lr)
            print('New lr:', param_group['lr'])
    elif g_conf.LEARNING_RATE_SCHEDULE == 'warmup_cooldown':
        assert None not in (iteration, total_iterations), 'Iteration and total iterations must be provided for warmup_cooldown schedule'
        warmup = max(0.0, min(1.0, g_conf.LEARNING_RATE_POLICY['warmup']))  # Ensure warmup is in [0, 1]
        warmup = warmup * total_iterations
        if iteration < g_conf.LEARNING_RATE_POLICY['warmup']:
            new_lr = iteration * (g_conf.LEARNING_RATE - min_lr) / g_conf.LEARNING_RATE_POLICY['warmup'] + min_lr
        else:
            new_lr = torch.cos(torch.pi * (iteration - g_conf.LEARNING_RATE_POLICY['warmup']) / (total_iterations - warmup))
            new_lr *= (g_conf.LEARNING_RATE - min_lr) / 2
            new_lr += (g_conf.LEARNING_RATE + min_lr) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    else:
        raise NotImplementedError('Not found learning rate policy!')
