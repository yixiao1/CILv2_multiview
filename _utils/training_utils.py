import os
import re
import random
import torch
import glob
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
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


class DataParallelDPPWrapper(DDP):
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

def extract_epoch_from_filename(filename):
    """
    Extract epoch number from checkpoint filename.
    Supports various naming patterns like:
    - CIL_multiview_20.pth
    - CILv2_multiview_attention_15_182081.pth
    - model_epoch_05_batch_1000.pth
    
    Returns the first number found after common prefixes.
    """
    basename = os.path.basename(filename)
    
    # Remove the file extension
    name_without_ext = os.path.splitext(basename)[0]
    
    # Find all complete numbers in the filename
    numbers = re.findall(r'\d+', name_without_ext)
    
    if not numbers:
        return None
    
    # Convert to integers
    numbers = [int(num) for num in numbers]
    
    # Common patterns to identify epoch numbers
    # Try to find epoch after common keywords first
    epoch_patterns = [
        r'epoch[_\-]?(\d+)',
        r'ckpt[_\-]?(\d+)',
        r'checkpoint[_\-]?(\d+)'
    ]
    
    for pattern in epoch_patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # For your specific naming pattern: CILv2_multiview_attention_15_182081.pth
    # The epoch number comes after the main part and before any batch/step numbers
    # We'll look for patterns where there are underscores separating numbers
    
    # Split by underscores and find number segments
    parts = name_without_ext.split('_')
    
    # Look for the first standalone number part (not embedded in text)
    for part in parts:
        if part.isdigit():
            return int(part)
    
    # If no standalone number found, return the first number
    # This handles cases like CIL_multiview_20.pth where 20 might be part of a larger string
    return numbers[0]


def check_saved_checkpoints(checkpoints_path: str, ckpt_number: int = None):
    """
    Returns the checkpoint file from checkpoints_path. If no ckpt_number 
    is specified, it returns the last found checkpoint file.
    
    Parameters:
        checkpoints_path (str): Path where the checkpoints are saved at.
        ckpt_number (int, optional): The specific checkpoint number to return.

    Returns:
        str: The requested checkpoint filename path.
    """
    if not os.path.exists(checkpoints_path):
        print('Given checkpoints path does not exist!')
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoints_path, '*.pth'))
    if not checkpoints:
        print('No checkpoint files found!')
        return None
    
    # Create list of (filename, epoch_number) tuples
    checkpoint_epochs = []
    for checkpoint in checkpoints:
        epoch = extract_epoch_from_filename(checkpoint)
        if epoch is not None:
            checkpoint_epochs.append((checkpoint, epoch))
    
    if not checkpoint_epochs:
        print('No valid epoch numbers found in checkpoint filenames!')
        return None
    
    # Sort by epoch number
    checkpoint_epochs.sort(key=lambda x: x[1])
    
    if ckpt_number is None:
        # Return the checkpoint with the highest epoch number
        return checkpoint_epochs[-1][0]
    else:
        # Find checkpoint with matching epoch number
        for checkpoint, epoch in checkpoint_epochs:
            if epoch == ckpt_number:
                return checkpoint
        
        print(f'No saved checkpoints found with epoch number {ckpt_number}!')
        available_epochs = [epoch for _, epoch in checkpoint_epochs]
        print(f'Available epochs: {sorted(available_epochs)}')
        return None


def check_saved_checkpoints_in_total(checkpoints_path):
    """
    Returns all checkpoint files sorted by epoch number.
    
    Parameters:
        checkpoints_path (str): Path where the checkpoints are saved at.
        
    Returns:
        list: List of checkpoint filenames sorted by epoch number, or None if no checkpoints found.
    """
    if not os.path.exists(checkpoints_path):
        print('Given checkpoints path does not exist!')
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoints_path, '*.pth'))
    if not checkpoints:
        return None
    
    # Create list of (filename, epoch_number) tuples
    checkpoint_epochs = []
    for checkpoint in checkpoints:
        epoch = extract_epoch_from_filename(checkpoint)
        if epoch is not None:
            checkpoint_epochs.append((checkpoint, epoch))
    
    if not checkpoint_epochs:
        return None
    
    # Sort by epoch number and return just the filenames
    checkpoint_epochs.sort(key=lambda x: x[1])
    return [checkpoint for checkpoint, _ in checkpoint_epochs]



def check_saved_checkpoints(checkpoints_path: str, ckpt_number: int = None):
    """
    Returns the checkpoint file from checkpoints_path. If no ckpt_number 
    is specified, it returns the last found checkpoint file.
    
    Parameters:
        checkpoints_path (str): Path where the checkpoints are saved at.
        ckpt_number (int, optional): The specific checkpoint number to return.

    Returns:
        str: The requested checkpoint filename path.
    """
    if not os.path.exists(checkpoints_path):
        print('Given checkpoints path does not exist!')
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoints_path, '*.pth'))
    if not checkpoints:
        print('No checkpoint files found!')
        return None
    
    # Create list of (filename, epoch_number) tuples
    checkpoint_epochs = []
    for checkpoint in checkpoints:
        epoch = extract_epoch_from_filename(checkpoint)
        if epoch is not None:
            checkpoint_epochs.append((checkpoint, epoch))
    
    if not checkpoint_epochs:
        print('No valid epoch numbers found in checkpoint filenames!')
        return None
    
    # Sort by epoch number
    checkpoint_epochs.sort(key=lambda x: x[1])
    
    if ckpt_number is None:
        # Return the checkpoint with the highest epoch number
        return checkpoint_epochs[-1][0]
    else:
        # Find checkpoint with matching epoch number
        for checkpoint, epoch in checkpoint_epochs:
            if epoch == ckpt_number:
                return checkpoint
        
        print(f'No saved checkpoints found with epoch number {ckpt_number}!')
        available_epochs = [epoch for _, epoch in checkpoint_epochs]
        print(f'Available epochs: {sorted(available_epochs)}')
        return None


def check_saved_checkpoints_in_total(checkpoints_path):
    """
    Returns all checkpoint files sorted by epoch number.
    
    Parameters:
        checkpoints_path (str): Path where the checkpoints are saved at.
        
    Returns:
        list: List of checkpoint filenames sorted by epoch number, or None if no checkpoints found.
    """
    if not os.path.exists(checkpoints_path):
        print('Given checkpoints path does not exist!')
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoints_path, '*.pth'))
    if not checkpoints:
        return None
    
    # Create list of (filename, epoch_number) tuples
    checkpoint_epochs = []
    for checkpoint in checkpoints:
        epoch = extract_epoch_from_filename(checkpoint)
        if epoch is not None:
            checkpoint_epochs.append((checkpoint, epoch))
    
    if not checkpoint_epochs:
        return None
    
    # Sort by epoch number and return just the filenames
    checkpoint_epochs.sort(key=lambda x: x[1])
    return [checkpoint for checkpoint, _ in checkpoint_epochs]


def update_learning_rate(optimizer: torch.optim.Optimizer,
                         iteration: int = None,
                         total_iterations: int = None,
                         min_lr: float = g_conf.LEARNING_RATE_MINIMUM) -> None:
    """ Adjusts the learning rate based on the schedule """
    if g_conf.LEARNING_RATE_SCHEDULE == 'step':
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
        if iteration < warmup:
            new_lr = iteration * (g_conf.LEARNING_RATE - min_lr) / warmup + min_lr
        else:
            new_lr = np.cos(np.pi * (iteration - warmup) / (total_iterations - warmup))
            new_lr *= (g_conf.LEARNING_RATE - min_lr) / 2
            new_lr += (g_conf.LEARNING_RATE + min_lr) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    elif g_conf.LEARNING_RATE_SCHEDULE == 'constant':
        pass

    else:
        raise NotImplementedError('Not found learning rate policy!')
