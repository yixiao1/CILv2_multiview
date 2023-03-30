import numpy as np
from pytorch_grad_cam.pytorch_grad_cam.base_cam import BaseCAM
import torch

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor_list,
                        target_layer,
                        target_category,
                        activations,
                        grads):  # grads size [B, seq_len, hidden_dim]
        return np.mean(grads, axis=2)  # TODO: axis=(2, 3) for attention maps