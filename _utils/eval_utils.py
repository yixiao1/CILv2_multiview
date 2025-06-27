import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from network.models.architectures.CIL_multiview.CIL_multiview import CIL_multiview
import os
import json
from configs import g_conf, set_type_of_process, merge_with_yaml
from _utils.training_utils import check_saved_checkpoints
import torch
from typing import OrderedDict, Union, Tuple, List
from PIL import Image
from einops import rearrange
from dataloaders.transforms import canbus_normalization, train_transform


# load the model ============================

def load_model_from_checkpoint(model: CIL_multiview, 
                               checkpoint_path: str, 
                               checkpoint_number: int) -> CIL_multiview:
    """
    Load model from checkpoint, handling various prefix patterns from multi-GPU training.
    
    Handles these patterns:
    - Direct keys: "encoder_embedding_perception.conv1.weight"
    - _model prefix: "_model.encoder_embedding_perception.conv1.weight" 
    - module._model prefix: "module._model.encoder_embedding_perception.conv1.weight"
    - module prefix: "module.encoder_embedding_perception.conv1.weight"
    """
    checkpoint = check_saved_checkpoints(checkpoint_path, checkpoint_number)  # works even if checkpoint_number is None
    checkpoint = torch.load(checkpoint)

    # Get the state dict from checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the entire checkpoint is the state dict
        state_dict = checkpoint

    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_key = k
        
        # Handle different prefix patterns
        if k.startswith('module._model.'):
            # Remove 'module._model.' prefix
            new_key = k[len('module._model.'):]
        elif k.startswith('model._model.'):
            new_key = k[len('model._model.'):]
        elif k.startswith('module.'):
            # Remove 'module.' prefix
            new_key = k[len('module.'):]
        elif k.startswith('_model.'):
            # Remove '_model.' prefix
            new_key = k[len('_model.'):]
        # If no prefix matches, keep the original key
        
        new_state_dict[new_key] = v
        
    new_state_dict = OrderedDict(new_state_dict)
    
    # Remove PE
    for k, v in new_state_dict.items():
        if k.startswith('positional_encoding'):
            new_state_dict[k] = torch.zeros(1, 0, 512)

    model.load_state_dict(new_state_dict)
    return model

def load_config(exp_batch: str, exp_name: str) -> None:
    merge_with_yaml(os.path.join('configs', exp_batch, f'{exp_name}.yaml'))


def get_positional_embedding(exp_batch: str, exp_name: str, checkpoint_number: int = None) -> torch.Tensor:
    load_config(exp_batch, exp_name)
    model = CIL_multiview(g_conf.MODEL_CONFIGURATION, average_attn_weights=False).to('cuda')
    checkpoint_path = os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results', 
                               g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME, 'checkpoints')
    load_model_from_checkpoint(model=model, checkpoint_path=checkpoint_path, checkpoint_number=checkpoint_number)

    return model.positional_encoding


# load the data ==============================

def open_image(dataset_root_path: Union[str, os.PathLike],
               img_name: str) -> Image.Image:
    """
    Open and return a PIL.Image.Image object from the given image name.
    """
    img = Image.open(os.path.join(dataset_root_path, img_name))
    if 'virtual_attention' in img_name:
        return img.convert('L')
    return img.convert('RGB')

def get_single_data_point(data_root_dir: Union[str, os.PathLike],
                          rgb_img_name: str,
                          canbus_json_name: str,
                          data_number: int, 
                          synth_att_name: str = None,
                          override_speed: float = None,
                          override_direction: float = None) -> dict:
    """
    
    """

    img_paths_dict = {
        f'{rgb_img_name}_central': open_image(data_root_dir, f'{rgb_img_name}_central{data_number:06d}.png'),
        f'{rgb_img_name}_left': open_image(data_root_dir, f'{rgb_img_name}_left{data_number:06d}.png'),
        f'{rgb_img_name}_right': open_image(data_root_dir, f'{rgb_img_name}_right{data_number:06d}.png'),
    }

    if synth_att_name is not None:
        img_paths_dict.update({
            f'{synth_att_name}_central_': open_image(data_root_dir, f'{synth_att_name}_central_{data_number:06d}.jpg'),
            f'{synth_att_name}_left_': open_image(data_root_dir, f'{synth_att_name}_left_{data_number:06d}.jpg'),
            f'{synth_att_name}_right_': open_image(data_root_dir, f'{synth_att_name}_right_{data_number:06d}.jpg'),
        })

    canbus_path = os.path.join(data_root_dir, f'{canbus_json_name}{data_number:06d}.json')

    datapoint = {}
    datapoint['can_bus'] = dict()
    f = open(canbus_path, 'r')
    canbus_data = json.loads(f.read())
    
    if override_speed is not None:
        canbus_data['speed'] = override_speed
    if override_direction is not None:
        canbus_data['direction'] = override_direction

    for value in g_conf.TARGETS + g_conf.OTHER_INPUTS:
        datapoint['can_bus'][value] = canbus_data[value]
    datapoint['can_bus'] = canbus_normalization(datapoint['can_bus'], g_conf.DATA_NORMALIZATION)
    for camera_type, img_paths in img_paths_dict.items():
        datapoint[camera_type] = img_paths

    return train_transform(datapoint, tuple(g_conf.IMAGE_SHAPE))


def model_forward(model: CIL_multiview, 
                  data: dict,
                  last_encoder_state: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    Get the intermediate features of the CIL_multiview model.
        return_attentions (bool): return the attention weights of the Transformer Encoder
    """
    src_images = [data[camera_type] for camera_type in g_conf.DATA_USED if 'rgb' in camera_type]
    src_directions = [torch.tensor([data['can_bus']['direction']]).float()]
    src_s = [torch.tensor([data['can_bus']['speed']]).float()]

    x = torch.stack(src_images).to('cuda').squeeze(1)
    d = src_directions[-1].to('cuda')
    s = src_s[-1].to('cuda')

    e_p, resnet_inter = model.encoder_embedding_perception(x)

    e_p = rearrange(e_p, 'b dim h w -> 1 (b h w) dim')

    if model.num_register_tokens > 0:
        e_p = torch.cat([model.register_tokens, e_p], dim=1)

    e_d = model.command(d).unsqueeze(1)
    e_s = model.speed(s).unsqueeze(0).unsqueeze(0)

    e_p = e_p + e_d + e_s
    e_p = e_p + model.positional_encoding

    in_memory, attn_weights = model.tx_encoder(e_p)

    action_output, _, _ = model.action_prediction(in_memory, cam=len(src_images))
    if last_encoder_state:
        return action_output, resnet_inter, attn_weights, in_memory
    return action_output, resnet_inter, attn_weights

# ============== Plotting utils =====================

import cv2
import torchvision.transforms.functional as TF
import numpy as np 

from einops import rearrange
import torch.nn.functional as F
from _utils import utils

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}" #for \text command

def plot_attention_weights_cossim(attn_weights: list) -> None:
    """
    Plot the pairwise cosine similarities between the attention maps of the heads in the Transformer Encoder.
    """
    # Loop over each layer

    for layer in range(len(attn_weights)):
        last_layer_maps = attn_weights[layer].squeeze(0)  # Shape is [4, 300, 300]

        # Flatten each head's attention map
        flattened_maps = last_layer_maps.view(4, -1)  # Shape is [4, 90000]

        # Compute pairwise cosine similarity
        cosine_similarities = torch.ones((4, 4))
        for i in range(4):
            for j in range(i+1, 4):  # Only fill above the diagonal
                cosine_similarities[i, j] = F.cosine_similarity(flattened_maps[i].unsqueeze(0), flattened_maps[j].unsqueeze(0))

        # Convert to numpy for plotting
        cosine_similarities_np = 1 - cosine_similarities.detach().cpu().numpy()

        # Set up the matplotlib figure
        plt.figure(figsize=(8, 6))

        # Draw the heatmap with the mask
        sns.heatmap(cosine_similarities_np, annot=True, fmt=".3f", cmap='inferno',
                    cbar_kws={"shrink": .8}, square=True, linewidths=.5)

        # Add labels and a title for clarity
        plt.title(r'$\texttt{cossim}$' + f' Between Heads - Layer Idx. {layer},' + r'$\mathcal{L}=$' + f'{cosine_similarities_np.sum():.2f}')
        plt.xlabel('Head Index')
        plt.ylabel('Head Index')

        # Show the plot
        plt.show()


def plot_interhead_cossim_curves(attn_weights: list):
    """
    Plot the pairwise cosine similarities between the attention maps of the heads 
    in the Transformer Encoder. Plot the average of the four heads per layer, with
    the standard deviation.
    """
    # Loop over each layer

    plt.figure(figsize=(8, 6))
    for layer in range(len(attn_weights)):
        last_layer_maps = attn_weights[layer].squeeze(0)  # Shape is [4, 300, 300]

        # Flatten each head's attention map
        flattened_maps = last_layer_maps.view(4, -1)  # Shape is [4, 90000]

        # Compute pairwise cosine similarity
        cosine_similarities = torch.zeros(int(4*3//2))
        idx = 0
        for i in range(4):
            for j in range(i+1, 4):  # Only fill above the diagonal
                cosine_similarities[idx] = F.cosine_similarity(flattened_maps[i].unsqueeze(0), flattened_maps[j].unsqueeze(0))
                idx += 1
                
        # Convert to numpy for plotting
        cosine_similarities_np = 1 - cosine_similarities.detach().cpu().numpy()

        # Compute the average and standard deviation
        avg = cosine_similarities_np.mean()
        std = cosine_similarities_np.std()

        # Plot the average and standard deviation
        plt.errorbar(layer, avg, yerr=std, fmt='o', color='blue')

    # Add labels and a title for clarity
    plt.title(r'$\texttt{cossim}$' + ' Between Heads - Average Per Layer')
    plt.xlabel('Layer Index')
    plt.ylabel(r'$\texttt{cossim}$')

    # Show the plot
    plt.show()


def plot_pure_attention_weights_heatmap(attn_weights: list) -> None:
    """
    Plot the attention weights of the Transformer Encoder.
    """
    num_heads = attn_weights[0].shape[1]
    num_layers = len(attn_weights)

    fig, ax = plt.subplots(num_layers, num_heads + 1, figsize=(5*(num_heads+1), 5*num_layers))
    for idx, attn in enumerate(attn_weights):
        for head in range(num_heads + 1):
            # Only set the title in the first row and first column; the rest are redundant
            if idx == 0:
                ax[idx, head].set_title(f'Head {head + 1}', fontsize=24, weight="bold")
            if head == 0:
                ax[idx, head].set_ylabel(f'Layer {idx + 1}', fontsize=24, weight="bold")
            if head == num_heads:
                if idx == 0:
                    ax[idx, head].set_title('Average Att. Weight', fontsize=24, weight="bold")
                ax[idx, head].imshow(attn[0].mean(dim=0).cpu().detach().numpy(), cmap='inferno')

            else:
                ax[idx, head].imshow(attn[0, head].cpu().detach().numpy(), cmap='inferno')
            ax[idx, head].set_xticks([])
            ax[idx, head].set_yticks([])

    plt.tight_layout()

    # Adding the rectangle patch for the last column
    bbox = ax[0, num_heads].get_position()  # Get the position of the last column's first subplot
    rect = mpl.patches.Rectangle((bbox.x0, 0), bbox.width, 1, transform=fig.transFigure, color='hotpink', zorder=-1)
    fig.patches.append(rect)

    plt.show()


def plot_attention_weights_on_images(attn_weights: list, src_images: list, model: CIL_multiview) -> None:
    einops_reshape_str = '(h w S cam) -> h (w S cam)' if (g_conf.ATTENTION_LOSS or g_conf.MHA_ATTENTION_COSSIM_LOSS) else '(S cam h w) -> h (S cam w)'

    img_cat = torch.cat(src_images, dim=2)
    img_cat = TF.normalize(img_cat, [-0.485/0.229, -0.456/0.224, -0.406/0.255], [1/0.229, 1/0.224, 1/0.255])
    img_cat = torch.clamp(img_cat, 0, 1)
    img_cat = img_cat.cpu().numpy().transpose(1, 2, 0)

    num_heads = attn_weights[0].shape[1]
    num_layers = len(attn_weights)

    fig, ax = plt.subplots(num_layers, num_heads + 1, figsize=(6*(num_heads+1), 2*num_layers))
    for idx, attn in enumerate(attn_weights):
        for head in range(num_heads + 1):
            ax[idx, head].imshow(img_cat)
            # Only set the title in the first row and first column; the rest are redundant
            if idx == 0:
                ax[idx, head].set_title(f'Head {head + 1}', fontsize=24)
            if head == 0:
                ax[idx, head].set_ylabel(f'Layer {idx + 1}', fontsize=24)
            if head == num_heads:
                if idx == 0:
                    ax[idx, head].set_title('Average Att. Weight', fontsize=24)

                atten = attn[0].mean(dim=(0, 1))
                atten = rearrange(atten, einops_reshape_str, S=1, cam=3, h=model.res_out_h).cpu().detach().numpy()
                atten = cv2.resize(atten, (img_cat.shape[1], img_cat.shape[0]), interpolation=cv2.INTER_LINEAR)
                atten = utils.min_max_norm(atten)
                ax[idx, head].imshow(atten, cmap='inferno', alpha=0.7, aspect='auto')

            else:
                atten = attn[0, head].squeeze().mean(0)
                atten = rearrange(atten, einops_reshape_str, S=1, cam=3, h=model.res_out_h).cpu().detach().numpy()
                atten = cv2.resize(atten, (img_cat.shape[1], img_cat.shape[0]), interpolation=cv2.INTER_LINEAR)
                atten = utils.min_max_norm(atten)
                ax[idx, head].imshow(atten, cmap='inferno', alpha=0.7, aspect='auto')
            ax[idx, head].set_xticks([])
            ax[idx, head].set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0, top=1.0, bottom=0.0)  # Reduce space between columns

    plt.show()


@torch.no_grad()
def patch_cos_sim(positional_embedding: torch.Tensor, idx_i: int, idx_j: int, rows: int = 10, cols: int = 30):
    """ Return a MxN matrix of cosine similarity between patch at (idx_i, idx_j) and the rest of the patches """
    positional_embedding = rearrange(positional_embedding, '1 (h w) d -> h w d' if g_conf.ATTENTION_LOSS else '1 (w h) d -> h w d', h=rows, w=cols)
    positional_embedding = positional_embedding.cpu()
    nrows, ncols, _ = positional_embedding.shape
    cossim_patches = []
    for i in range(nrows):
        for j in range(ncols):
            cossim_patches.append(F.cosine_similarity(positional_embedding[idx_i, idx_j], positional_embedding[i, j], dim=0))

    return np.array(cossim_patches).reshape(nrows, ncols)

# Function to plot cosine similarity for a specific camera
def plot_cosine_similarity_for_camera(positional_embeddings, start_col, end_col, camera_name):
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(30, 10))
    for i in range(10):
        for j in range(start_col, end_col):
            csim = patch_cos_sim(positional_embeddings, i, j)
            axes[i, j - start_col].set_axis_off()
            axes[i, j - start_col].imshow(csim, cmap='jet', interpolation='nearest')
    fig.suptitle(f'CosSim Positional Embeddings - {camera_name}', fontsize=32)
    plt.tight_layout()
    plt.show()

def plot_all_cosine_similarities_cameras(positional_embeddings):
    plot_cosine_similarity_for_camera(positional_embeddings, 0, 10, 'Left Camera')
    plot_cosine_similarity_for_camera(positional_embeddings, 10, 20, 'Central Camera')
    plot_cosine_similarity_for_camera(positional_embeddings, 20, 30, 'Right Camera')

def get_average_cossim_per_camera(positional_embeddings: torch.Tensor, 
                                  nrows: int = 10, ncols_percam: int = 10, 
                                  num_cameras: int = 3) -> np.ndarray:
    avg_cossim = np.zeros((num_cameras, nrows, num_cameras*ncols_percam))
    for cam in range(num_cameras):
        for i in range(nrows):
            for j in range(cam*ncols_percam, (cam+1)*ncols_percam):
                csim = patch_cos_sim(positional_embeddings, i, j)
                avg_cossim[cam] += csim / (ncols_percam * nrows)

    return avg_cossim