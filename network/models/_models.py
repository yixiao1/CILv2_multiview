import os
import importlib
import torch.nn as nn

from configs import g_conf
from network.loss import Loss
from dataloaders import make_data_loader
from _utils.evaluation import evaluation_on_model

from .architectures.CIL_multiview.evaluator import CIL_multiview_Evaluator


class CILv2_multiview_attention(nn.Module):
    def __init__(self, params, rank: int = 0):
        super(CILv2_multiview_attention, self).__init__()
        cil_model = importlib.import_module('network.models.architectures.CIL_multiview')
        cil_model = getattr(cil_model, g_conf.MODEL_TYPE)
        self.params = params
        self._model = cil_model(params, rank, average_attn_weights=False if (g_conf.MHA_ATTENTION_COSSIM_LOSS or g_conf.MHA_ATTENTION_LOSS) else True)
        self.resize_att_h, self.resize_att_w = self._model.resize_att_h, self._model.resize_att_w
        self.name = g_conf.MODEL_TYPE

        if g_conf.PROCESS_NAME == 'train_val':
            self._current_iteration = 0
            self._done_epoch = 0
            self._criterion = Loss(g_conf.LOSS)
            self._train_loader, self._val_loaders = \
                make_data_loader(self.name, os.environ["DATASET_PATH"], g_conf.TRAIN_DATASET_NAME, g_conf.BATCH_SIZE,
                                 g_conf.VALID_DATASET_NAME, g_conf.EVAL_BATCH_SIZE, rank=params['rank'],
                                 num_process=params['num_process'], res_attr=(self.resize_att_w, self.resize_att_h))


            if rank == 0:
                if len(self._train_loader.dataset) == 0:
                    raise ValueError(f"Empty training dataset: {g_conf.TRAIN_DATASET_NAME}. Please check that the dataset path and the DATASET_PATH environment variable are correct!")

                if len(self._val_loaders) == 0:
                    raise ValueError(f"Empty validation dataset: {g_conf.VALID_DATASET_NAME}. Please check that the dataset path and the DATASET_PATH environment variable are correct!")

                print('\n================================= Dataset Info ========================================')
                print(f"\nUsing {len(g_conf.TRAIN_DATASET_NAME)} Training Dataset:")
                print(f'   - {g_conf.TRAIN_DATASET_NAME}: Total amount={len(self._train_loader.dataset)}')

                print(f"Using {len(self._val_loaders)} Validation Dataset:")
                for val_loader in self._val_loaders:
                    print(f'   - {val_loader.dataset.dataset_name}: {len(val_loader.dataset)}')
                print('')
                print('=======================================================================================')
                print('')

            self._dataloader_iter = iter(self._get_dataloader())

        elif g_conf.PROCESS_NAME == 'train_only':
            self._current_iteration = 0
            self._done_epoch = 0
            self._criterion = Loss(g_conf.LOSS)
            self._train_loader, _ = \
                make_data_loader(self.name, os.environ["DATASET_PATH"], g_conf.TRAIN_DATASET_NAME, g_conf.BATCH_SIZE,
                                 g_conf.VALID_DATASET_NAME, g_conf.EVAL_BATCH_SIZE, rank=params['rank'],
                                 num_process=params['num_process'], res_attr=(self.resize_att_w, self.resize_att_h))

            if len(self._train_loader.dataset) == 0:
                raise ValueError(f"Empty training dataset: {g_conf.TRAIN_DATASET_NAME}. Please check that the dataset path and the DATASET_PATH environment variable are correct!")

            if rank == 0:
                print('\n================================= Dataset Info ========================================')
                print(f"\nUsing {len(g_conf.TRAIN_DATASET_NAME)} Training Dataset:")
                print(f'   - {g_conf.TRAIN_DATASET_NAME}: Total amount={len(self._train_loader.dataset)}')
                print('=======================================================================================')
                print('')

            self._dataloader_iter = iter(self._get_dataloader())

        elif g_conf.PROCESS_NAME == 'val_only':
            _, self._val_loaders = \
                make_data_loader(self.name, os.environ["DATASET_PATH"], g_conf.TRAIN_DATASET_NAME, g_conf.BATCH_SIZE,
                                 g_conf.VALID_DATASET_NAME, g_conf.EVAL_BATCH_SIZE)

            if len(self._val_loaders) == 0:
                raise ValueError(f"Empty validation dataset: {g_conf.VALID_DATASET_NAME}. Please check that the dataset path and the DATASET_PATH environment variable are correct!")

            if rank == 0:
                print("Using {} Validation Dataset:".format(str(len(self._val_loaders))))
                for val_loader in self._val_loaders:
                    print('   - '+val_loader.dataset.dataset_name+':', str(len(val_loader.dataset)))

        self._evaluator = CIL_multiview_Evaluator(self.name)

    def _get_dataloader(self):
        return self._train_loader

    def _eval(self, current_iteration, eval_epoch):
        self._current_iteration = current_iteration
        self._done_epoch = eval_epoch
        loader = self._val_loaders
        results_dict = evaluation_on_model(self, loader, self.name, self._evaluator,
                                           eval_iteration=self._current_iteration-1, eval_epoch=self._done_epoch)

        print(f'{self.name} evaluation results at iteration {self._current_iteration-1} / epoch {self._done_epoch}: {results_dict}')
        for dataset_name, _ in results_dict.items():
            results_dict[dataset_name]['iteration'] = (self._current_iteration-1)
            results_dict[dataset_name]['epoch'] = self._done_epoch
        return results_dict

    def forward(self, src_images, src_directions, src_speeds):
        return self._model.forward(src_images, src_directions, src_speeds)

    def forward_eval(self, src_images, src_directions, src_speeds, attn_rollout: bool = False, attn_refinement: bool = False):
        return self._model.forward_eval(src_images, src_directions, src_speeds, attn_rollout, attn_refinement)

    def loss(self, params):
        loss = self._criterion(params)
        return loss

    def __len__(self):
        return len(self._train_loader.dataset)
