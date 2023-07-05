
import os
import shutil

from configs._global import create_exp_path
from . import train_val, val

def execute_train_val(gpus_list, exp_batch, exp_alias, rank=0):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    if rank == 0:
        create_exp_path(os.environ['TRAINING_RESULTS_ROOT'], exp_batch, exp_alias)
    train_val.execute(gpus_list, exp_batch, exp_alias, rank=rank)


def execute_val(gpus_list, exp_batch, exp_alias, rank=0):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    if rank == 0:
        create_exp_path(os.environ['TRAINING_RESULTS_ROOT'],exp_batch, exp_alias)
    val.execute(gpus_list, exp_batch, exp_alias, rank=rank)


