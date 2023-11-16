import torch
import torch.nn.functional as F
import numpy as np

from configs import g_conf
from collections import OrderedDict
from typing import Union, List

import matplotlib.pyplot as plt
import os

from einops import rearrange


class CIL_multiview_Evaluator(object):
    """
    Evaluate
    """
    def __init__(self, name):
        self._action_batch_errors_mat = 0
        self.attentions_loss_pointwise = 0
        self._total_num = 0
        self._metrics = {}
        self.steers = []
        self.accelerations= []
        self.gt_steers = []
        self.gt_accelerations = []
        self.attentions = []
        self.gt_attentions = []

        self.name = name

    def reset(self):
        self._action_batch_errors_mat = 0
        self.attentions_loss_pointwise = 0
        self._total_num = 0
        self._metrics = {}
        self.steers = []
        self.accelerations= []
        self.gt_steers = []
        self.gt_accelerations = []
        self.attentions = []
        self.gt_attentions = []

    def process(self, 
                action_outputs: Union[torch.Tensor, tuple], 
                targets_action: torch.Tensor, 
                attention_outputs: torch.Tensor = None,
                targets_attention: torch.Tensor = None):
        """
        Compute the errors sum for the outputs and targets of the neural network in val dataset
        """
        if isinstance(action_outputs, (tuple, list)):
            action_outputs = action_outputs[0]
        B = action_outputs.shape[0]
        self._total_num += B
        action_outputs = action_outputs[:, -1, -len(g_conf.TARGETS):]
        self.steers += list(action_outputs[:, 0].detach().cpu().numpy())
        self.accelerations += list(action_outputs[:, 1].detach().cpu().numpy())
        self.gt_steers += list(targets_action[-1][:, 0].detach().cpu().numpy())
        self.gt_accelerations += list(targets_action[-1][:, 1].detach().cpu().numpy())
        if None not in (attention_outputs, targets_attention):
            attention_outputs = rearrange(attention_outputs, 'B ... -> B (...)')
            self.attentions.append(attention_outputs)
            self.gt_attentions.append(targets_attention)
            self.attentions_loss_pointwise += F.kl_div((attention_outputs + 1e-12).log(), targets_attention, reduction='sum')  
        actions_loss_mat_normalized = torch.clip(action_outputs, -1, 1) - targets_action[-1]  # (B, len(g_conf.TARGETS))

        # unnormalize the outputs and targets to compute actual error
        if g_conf.ACCELERATION_AS_ACTION:
            self._action_batch_errors_mat += torch.abs(actions_loss_mat_normalized)  # [-1, 1]

        else:
            pass

    def evaluate(self, current_epoch, dataset_name):
        self.metrics_compute(self._action_batch_errors_mat, self.attentions_loss_pointwise)
        results = OrderedDict({self.name: self._metrics})

        def plot_and_save_results(data: Union[list, np.ndarray],
                                  ground_truth_data: Union[list, np.ndarray],
                                  save_name: str) -> None:
            """ Plot the data with the ground truth and save the figure """
            plt.figure()
            width, height = plt.gcf().get_size_inches()
            plt.gcf().set_size_inches([4.0 * width, height])
            plt.plot(range(len(ground_truth_data)), ground_truth_data, color='green')
            plt.plot(range(len(data)), data, color='blue')
            plt.ylim([-1.2, 1.2])
            plt.xlabel('Frame Id')
            plt.ylabel('')
            plt.savefig(os.path.join(g_conf.EXP_SAVE_PATH, f'{save_name}.jpg'))
            plt.close()

        # Plot the accelerations w.r.t. the ground truth on the validation set
        plot_and_save_results(self.accelerations, self.gt_accelerations, f'acc_{dataset_name}_epoch{current_epoch}')

        # Plot the steering angles w.r.t. the ground truth on the validation set
        plot_and_save_results(self.steers, self.gt_steers, f'steer_{dataset_name}_epoch{current_epoch}')

        def save_frame_ids_different_signs(data: Union[list, np.ndarray],
                                           ground_truth_data: Union[list, np.ndarray],
                                           save_name: str) -> None:
            """ Save the frame ids when the sign of the data is different from the ground truth as a csv file """
            # First, if the file doesn't exist, create it, with the first line reading "epoch" and "frame_id"
            sdir = os.path.join(g_conf.EXP_SAVE_PATH, f'{save_name}.txt')
            if not os.path.exists(sdir):
                with open(sdir, 'w') as f:
                    f.write('epoch, total_different, frame_id')

            # With this file, append the current epoch and the frame ids where the sign of the acceleration is different
            # from the ground truth
            with open(sdir, 'a') as f:
                different_frames_idx = np.where(np.sign(data) != np.sign(ground_truth_data))[0]
                f.write(f'\n{current_epoch}, {len(different_frames_idx)}, {different_frames_idx.tolist()}')

        # Save the frame ids where the sign of the acceleration is different from the ground truth
        save_frame_ids_different_signs(self.accelerations, self.gt_accelerations, f'acc-sign_{dataset_name}')

        # Ibidem for steer sign
        save_frame_ids_different_signs(self.steers, self.gt_steers, f'steer-sign_{dataset_name}')

        return results

    def metrics_compute(self, action_errors_mat: torch.Tensor, att_loss_pointwise: torch.Tensor = torch.tensor(0.0)) -> None:

        self._metrics.update({'MAE_steer': torch.sum(action_errors_mat, 0)[0] / self._total_num})
        if g_conf.ACCELERATION_AS_ACTION:
            self._metrics.update({'MAE_acceleration': torch.sum(action_errors_mat, 0)[1] / self._total_num})
        else:
            pass

        if g_conf.ATTENTION_LOSS:
            self._metrics.update({'MKLdiv_attention': att_loss_pointwise / self._total_num})
        att_loss = att_loss_pointwise.sum() if g_conf.ATTENTION_LOSS else torch.tensor(0.0)
        self._metrics.update({'MAE': (att_loss + torch.sum(action_errors_mat)) / self._total_num})