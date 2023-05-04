from __future__ import unicode_literals
from typing import Union

import os
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from .tensorboard_logger import Logger
import cv2

# We keep the file names saved here in the glogger to avoid including global
TRAIN_IMAGE_LOG_FREQUENCY = 1
TRAIN_LOG_FREQUENCY = 1
tl = ''


def create_log(save_full_path: Union[str, os.PathLike],
               train_log_frequency: int = 1,
               train_image_log_frequency: int = 15) -> None:
               #eval_log_frequency=1, image_log_frequency=15):
    """
    Arguments
        save_full_path: the full path to save the tensorboard logs
        log_frequency: frequency to log values
        image_log_frequency: frequency to log images
    """
    global tl
    global TRAIN_LOG_FREQUENCY
    global TRAIN_IMAGE_LOG_FREQUENCY

    TRAIN_LOG_FREQUENCY = train_log_frequency
    TRAIN_IMAGE_LOG_FREQUENCY = train_image_log_frequency
    tl = Logger(os.path.join(save_full_path, 'tensorboard_logs'))


def add_scalar(tag: str, value: float, iteration: int = None) -> None:

    """
        For raw outputs logging on tensorboard.
    """

    if iteration is not None:
        if iteration % TRAIN_LOG_FREQUENCY == 0:
            tl.scalar_summary(tag, value, iteration)
    else:
        raise ValueError('iteration is not supposed to be None')


def add_image(tag: str, images: torch.Tensor, num_images: int = 3, iteration: int = None) -> None:
    # Add the image to a log, the monitor is the module responsible by checking this
    # and eventually put some images to tensorboard.
    if iteration is not None:
        if iteration % TRAIN_IMAGE_LOG_FREQUENCY == 0:
            images = images.view(-1, images.shape[1], images.shape[2], images.shape[3])[:num_images].cpu().data.numpy()
            new_images = []
            if images.shape[1] == 1:
                cmap = plt.get_cmap('inferno')
                for i in range(images.shape[0]):
                    this = cmap(images[i, 0])[:, :, :3]
                    new_images.append(this)
                images = np.array(new_images).transpose(0, 3, 1, 2)

            tl.image_summary(tag, images, iteration + 1)

    else:
        images = images.view(-1, images.shape[1], images.shape[2], images.shape[3])[:10].cpu().data.numpy()
        tl.image_summary(tag, images, iteration + 1)


# TODO
def add_vit_attention_maps_to_disk(process_type: str,
                                   model: nn.Module,
                                   source_input: Union[torch.Tensor, list],
                                   input_rgb_frames: Union[torch.Tensor, list],
                                   epoch: int,
                                   save_path: Union[str, os.PathLike] = None,
                                   batch_id=None) -> None:
    """Save the ViT Attention maps to the disk."""
    global TRAIN_IMAGE_LOG_FREQUENCY
    cmap = plt.get_cmap('inferno')

    ## For saving training attention maps of the backbone
    if process_type == 'Train':
        pass

    ## For saving validation attention maps of the backbone
    elif process_type == 'Valid':
        S = len(source_input[0])  # how many frames are in the sequence
        cam_num = len(source_input[0][0])  # number of views/cameras (default: 3)
        _, C, H, W = source_input[0][0][0].shape

        attn_weights = model._model.forward_eval(*source_input)[1]

        # Get the steering [STR] attention map
        grayscale_cam_str = attn_weights[:, 0, :, :].detach().cpu().numpy()  # [S*cam, H, W]; STR token
        grayscale_cam_str = grayscale_cam_str.transpose(1, 2, 0)  # [H, W, S*cam]
        grayscale_cam_str = cv2.resize(grayscale_cam_str, (H, W), interpolation=cv2.INTER_AREA)  # cv2 thinks it has multiple channels
        grayscale_cam_str = grayscale_cam_str.transpose(2, 0, 1)  # [S*cam, H, W]
        grayscale_cam_str = grayscale_cam_str.reshape((S, cam_num, H, W))

        # Get the acceleration [ACC] attention map
        grayscale_cam_acc = attn_weights[:, 1, :, :].detach().cpu().numpy()  # [S*cam, H, W]; ACC token
        grayscale_cam_acc = grayscale_cam_acc.transpose(1, 2, 0)  # [H, W, S*cam]
        grayscale_cam_acc = cv2.resize(grayscale_cam_acc, (H, W), interpolation=cv2.INTER_AREA)  # cv2 thinks it has multiple channels
        grayscale_cam_acc = grayscale_cam_acc.transpose(2, 0, 1)  # [S*cam, H, W]
        grayscale_cam_acc = grayscale_cam_acc.reshape((S, cam_num, H, W))

        grayscale_cam = [grayscale_cam_str, grayscale_cam_acc]  # 2 * [S, cam, H, W]
        cam_names = ['STR', 'ACC']
        for idx, gcam in enumerate(grayscale_cam):
            Seq = []
            for s in range(S):
                cams = []
                for cam_id in range(cam_num):
                    att = gcam[s, cam_id, :]
                    cmap_att = np.delete(cmap(att), 3, 2)
                    cmap_att = Image.fromarray((cmap_att * 255).astype(np.uint8))
                    # cams.append(cmap_att)
                    cams.append(Image.blend(Image.fromarray(((input_rgb_frames[s][cam_id]).transpose(1, 2, 0) * 255).astype(np.uint8)), cmap_att, 0.5))
                Seq.append(np.concatenate(cams, 1))
            current_att = np.concatenate(Seq, 0)

            if save_path:
                if not os.path.exists(os.path.join(save_path, str(epoch), '-1')):
                    os.makedirs(os.path.join(save_path, str(epoch), '-1'))

                # we save the wanted layers of the backbone to the disk
                current_att = Image.fromarray(current_att)
                current_att.save(os.path.join(save_path, str(epoch), '-1', str(batch_id) + f'{cam_names[idx]}.jpg'))

            else:
                raise RuntimeError('You need to set the save_path')


def add_gradCAM_attentions_to_disk(process_type, model, source_input, input_rgb_frames,
                                   epoch, save_path=None, batch_id=None):

    global TRAIN_IMAGE_LOG_FREQUENCY
    cmap = plt.get_cmap('jet')

    ## For saving training attention maps of the backbone
    if process_type == 'Train':
        pass

    ## For saving validation attention maps of the backbone
    elif process_type == 'Valid':
        pass
        # TODO: visualize the attention maps of the backbone!
        # S = len(source_input[0])
        # cam_num = len(source_input[0][0])
        # _, C, H, W = source_input[0][0][0].shape

        # target_layers = [model._model.encoder_embedding_perception.layer4[-1]]
        # cam = GradCAM(model=model._model, target_layers=target_layers)

        # with torch.enable_grad():
        #     grayscale_cam = cam(input_tensor_list=source_input)   # [S*cam, H, W]

        # grayscale_cam = grayscale_cam.reshape((S, cam_num, H, W))

        # Seq = []
        # for s in range(S):
        #     cams = []
        #     for cam_id in range(cam_num):
        #         att = grayscale_cam[s, cam_id, :]
        #         cmap_att = np.delete(cmap(att), 3, 2)
        #         cmap_att = Image.fromarray((cmap_att * 255).astype(np.uint8))
        #         # cams.append(cmap_att)
        #         cams.append(Image.blend(Image.fromarray(((input_rgb_frames[s][cam_id]).transpose(1, 2, 0) * 255).astype(np.uint8)), cmap_att, 0.5))
        #     Seq.append(np.concatenate(cams, 1))
        # current_att = np.concatenate(Seq, 0)

        # if save_path:
        #     if not os.path.exists(os.path.join(save_path, str(epoch), '-1')):
        #         os.makedirs(os.path.join(save_path, str(epoch), '-1'))

        #     # we save the wanted layers of the backbone to the disk
        #     current_att = Image.fromarray(current_att)
        #     current_att.save(os.path.join(save_path, str(epoch), '-1',
        #                      str(batch_id) +'.jpg'))

        # else:
        #     raise RuntimeError('You need to set the save_path')