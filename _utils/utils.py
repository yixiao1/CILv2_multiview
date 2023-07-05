import time
import torch
import glob
import os
import re
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt
from torch.nn import DataParallel

########################################################
### Color plate
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_SCARLET_RED_0 = (255, 0, 0)
COLOR_SKY_BLUE_0 = (0, 0, 255)
COLOR_GREEN_0 = (0, 255, 0)
COLOR_LIGHT_GRAY = (196, 196, 196)
COLOR_PINK = (255,19, 203)
COLOR_BUTTER_0 = (252, 233, 79)
COLOR_ORANGE_0 = (252, 175, 62)
COLOR_CHOCOLATE_0 = (233, 185, 110)
COLOR_CHAMELEON_0 = (138, 226, 52)
COLOR_PLUM_0 = (173, 127, 168)
COLOR_ALUMINIUM_0 = (238, 238, 236)

color_plate = {
    '0': COLOR_SCARLET_RED_0,
    '1': COLOR_GREEN_0,
    '2': COLOR_SKY_BLUE_0,
    '3': COLOR_ALUMINIUM_0,
    '4': COLOR_CHAMELEON_0,
    '5': COLOR_CHOCOLATE_0,
    '6': COLOR_LIGHT_GRAY,
    '7': COLOR_PLUM_0,
    '8': COLOR_ORANGE_0,
    '9': COLOR_BUTTER_0
}

########################################################


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]


def sort_nicely(l: list) -> None:
    l.sort(key=alphanum_key)


def experiment_log_path(experiment_path, dataset_name):
    # WARNING if the path exist without checkpoints it breaks
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    return os.path.join(experiment_path, dataset_name + '_result.csv')


class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __len__(self):
        return len(self.module)


#@timeit
def print_train_info(log_frequency, final_epoch, batch_size, model,
                     time_start, acc_time, loss_data, steer_loss_data, acc_loss_data, brake_loss_data=None):

    epoch = model._current_iteration * batch_size / len(model)

    # Number of digits to print for the iteration and epoch counters
    digits_total_iters = int(np.log10(len(model))) + 1
    digits_total_epochs = int(np.log10(len(model) / batch_size)) + 1

    time_end = time.time()
    batch_time = time_end - time_start
    acc_time += batch_time

    if model._current_iteration % log_frequency == 0:

        rem_iterations = ((float(final_epoch) - epoch) * len(model)) / batch_size
        rem_time = rem_iterations / (log_frequency / acc_time)
        hours = int(rem_time / 60 / 60)
        minutes = int(rem_time / 60) - hours * 60
        seconds = int(rem_time) - hours * 60 * 60 - minutes * 60

        if brake_loss_data is not None:
            print(f"Training epoch {epoch:{digits_total_epochs}.2f}, iteration {model._current_iteration:{digits_total_iters}d}, Loss {loss_data:.4f}, "
                  f"Steer Loss {steer_loss_data:.4f}, Throttle Loss {acc_loss_data:.4f}, Brake Loss {brake_loss_data:.4f}, {log_frequency / acc_time:.2f} "
                  f"steps/s, ETA: {hours:0>2d}H:{minutes:0>2d}M:{seconds:0>2d}S")
        else:
            print(f"Training epoch {epoch:{digits_total_epochs}.2f}, iteration {model._current_iteration:{digits_total_iters}d}, Loss {loss_data:.4f}, "
                  f"Steer Loss {steer_loss_data:.4f}, Acc Loss {acc_loss_data:.4f}, {log_frequency / acc_time:.2f} steps/s, ETA: {hours:0>2d}H:{minutes:0>2d}M:{seconds:0>2d}S")

    return acc_time


def test_stop(number_of_data: int, iterated_data: int) -> bool:
    """ Check if the training should stop """
    return number_of_data != 0 and iterated_data >= number_of_data


def generate_specific_rows(filepath: Union[str, os.PathLike], row_indices: List[int] = None):
    with open(filepath) as f:
        # using enumerate to track line no.
        for i, line in enumerate(f):
            # if line no. is in the row index list, then return that line
            if i in row_indices:
                yield line.strip()


def read_results(result_file: Union[str, os.PathLike], metric: str = '') -> np.ndarray:
    head = np.genfromtxt(generate_specific_rows(result_file, row_indices=[0]),
                         delimiter=',', dtype='str')
    col = [h.strip() for h in list(head)].index(metric)
    res = np.loadtxt(result_file, delimiter=",", skiprows=1)
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)
    return res[:, col]


def draw_offline_evaluation_results(experiment_path: Union[str, os.PathLike],
                                    metrics_list: List[str],
                                    x_range: List[int] = None):
    for metric in metrics_list:
        print('drawing results graph for ', experiment_path, 'of', metric)
        results_files = glob.glob(os.path.join(experiment_path, '*.csv'))
        for results_file in results_files:
            plt.figure()
            output_path = os.path.join(experiment_path, results_file.split(os.sep)[-1].split('.')[-2]+ '_' + metric+'.jpg')
            results_list = read_results(results_file, metric=metric)
            epochs_list = read_results(results_file, metric='epoch')
            plt.ylabel(metric, fontsize=15)
            plt.plot(epochs_list, results_list)
            for i in range(len(results_list)):
                if results_list[i] == min(results_list):
                    plt.text(epochs_list[i], results_list[i], str(results_list[i]), color='blue',
                             fontweight='bold')
                    plt.plot(epochs_list[i], results_list[i], color='blue', marker='*')
            plt.xlabel('Epoch', fontsize=15)
            plt.xlim(left=x_range[0], right=x_range[-1])
            plt.title(results_file.split(os.sep)[-1].split('.')[-2])
            plt.savefig(output_path)
            plt.close()


def write_model_results(experiment_path, model_name, results_dict, acc_as_action=False):
    for dataset_name, results in results_dict.items():
        results_file_csv = experiment_log_path(experiment_path, dataset_name)
        new_row = ""
        # first row if file doest exist
        if not os.path.exists(results_file_csv):
            new_row += "iteration, epoch, "
            if acc_as_action:
                new_row += "MAE_steer, MAE_acceleration, MAE"
            else:
                new_row += "MAE_steer, MAE_throttle, MAE_brake, MAE"

            new_row += "\n"
        with open(results_file_csv, 'a') as f:
            new_row += "{}, {:.2f}, ".format(results['iteration'], results['epoch'])
            if acc_as_action:
                new_row += "{:.4f}, {:.4f}, {:.4f}".format(results[model_name]['MAE_steer'],
                                                           results[model_name]['MAE_acceleration'],
                                                           results[model_name]['MAE'])
            else:
                new_row += "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(results[model_name]['MAE_steer'], results[model_name]['MAE_throttle'],
                                                                   results[model_name]['MAE_brake'], results[model_name]['MAE'])

            new_row += "\n"
            f.write(new_row)
        print (" The results have been saved in: ", results_file_csv)


def eval_done(experiment_path, dataset_paths, epoch):
    results_files = glob.glob(os.path.join(experiment_path, '*.csv'))
    for dataset_path in dataset_paths:
        if os.path.join(experiment_path, dataset_path.split('/')[-1]+'_result.csv') not in results_files:
            return False
        else:
            epochs_list = read_results(os.path.join(experiment_path, dataset_path.split('/')[-1]+'_result.csv'), metric='epoch')
            if float(epoch) in epochs_list:
                return True
            else:
                return False


def is_result_better(experiment_path, model_name, dataset_name):
    results_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='MAE')
    iter_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='iteration')
    epoch_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='epoch')
    if len(results_list) == 1:  # There is just one result so we save the check sure.
        return True
    if results_list[-1] < min(results_list[:-1]):
        print("Result for {} at iteration {} / epoch {} is better than the previous one. SAVE".format(model_name, iter_list[-1], epoch_list[-1]))
        return True
    return False


def extract_targets(data, targets=[], ignore=[]):

    """
    Method used to get to know which positions from the dataset are the targets
    for this experiments
    Args:

    Returns:
        the float data that is actually targets

    Raises
        value error when the configuration set targets that didn't exist in metadata
    """

    targets_vec = []
    for target_name in targets:
        if target_name in ignore:
            continue
        targets_vec.append(data[target_name])

    return torch.stack(targets_vec, 1).float().squeeze()


def extract_other_inputs(data, other_inputs=[], ignore=[]):
    """
    Method used to get to know which positions from the dataset are the inputs
    for this experiments
    Args:

    Returns:
        the float data that is actually targets

    Raises
        value error when the configuration set targets that didn't exist in metadata
    """

    inputs_vec = []
    for input_name in other_inputs:
        if input_name in ignore:
            continue
        inputs_vec.append(data[input_name])
    return torch.stack(inputs_vec, 1).float()


def extract_commands(data):
    return torch.stack(data, 1).float()


def attn_rollout(attn_weights: List[torch.Tensor], layer: int = None) -> torch.Tensor:
    """ Perform attention rollout """
    att_map = torch.stack([attn_weights[i] for i in range(len(attn_weights))], dim=0)  # [L, C, S, S]
    num_layers, num_cams, S, _ = att_map.shape

    # Use the last layer if no layer is specified by the user
    layer = layer if layer is not None else num_layers
    layer = min(max(1, layer), num_layers)

    eye = torch.eye(S, device=att_map.device)  # [S, S])
    attn_weights_rollout_cams = torch.empty(layer, num_cams, S, S, device=att_map.device)

    for c in range(num_cams):
        attn_weights_rollout = [0.5 * att_map[0, c] + 0.5 * eye]  # list of 1 [S, S]

        for idx in range(1, layer):
            a = 0.5 * att_map[idx, c] + 0.5 * eye  # [S, S]
            a_tilde = torch.matmul(a, attn_weights_rollout[-1])  # [S, S]
            attn_weights_rollout.append(a_tilde)
        attn_weights_rollout_cams[:, c] = torch.stack(attn_weights_rollout)

    return attn_weights_rollout_cams
