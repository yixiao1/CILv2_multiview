from typing import Dict, List, Tuple, Generator, OrderedDict, Union, Optional
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import click
import pandas as pd

import json
import os
from PIL import Image
from einops import rearrange

from configs import g_conf, merge_with_yaml
from _utils.training_utils import check_saved_checkpoints
from _utils import utils
from dataloaders.transforms import canbus_normalization, ted_transform, decode_float_directions_to_str

from network.models.architectures.CIL_multiview.CIL_multiview import CIL_multiview


def load_model_from_checkpoint(model: CIL_multiview, 
                               checkpoint_path: str, 
                               checkpoint_number: int) -> CIL_multiview:
    checkpoint = check_saved_checkpoints(checkpoint_path, checkpoint_number)  # works even if checkpoint_number is None
    checkpoint = torch.load(checkpoint)

    new_state_dict = {}
    for k, v in checkpoint['model'].items():
        new_state_dict[k[7:]] = v

    new_state_dict = OrderedDict(new_state_dict)

    model.load_state_dict(new_state_dict)

    return model


def decode_batch_directions(directions_batch: torch.Tensor) -> List[float]:
    """Convert batch of one-hot directions to float values."""
    decoded = []
    for one_hot in directions_batch:
        one_hot_list = one_hot.tolist()
        index = one_hot_list.index(max(one_hot_list))
        # Use the direction mapping: 0->1.0, 1->2.0, 2->3.0, 3->4.0
        decoded.append(float(index + 1))
    return decoded


def model_forward_get_features(
        model: CIL_multiview, 
        data: dict,
        return_attentions: bool = False, 
        return_backbone_features: bool = False
    ) -> Union[list, torch.Tensor]:
    # Get the batch size
    batch_size = data['ar_resized_rgb_central'].shape[0]
    
    # Stack RGB images [batch, 3, h, w]
    src_images = [data[camera_type] for camera_type in g_conf.DATA_USED if 'rgb' in camera_type]
    x = torch.stack(src_images, dim=1)  # [batch, cam_num, 3, h, w]
    x = rearrange(x, 'B cam C H W -> (B cam) C H W')
    
    # Process directions
    d = data['cmd_fix_can_bus']['direction'].float()  # [batch, 4]
    # d = torch.tensor(directions, device='cuda').float().view(-1, 1)  # [batch, 1]
    
    # Process speeds
    s = data['cmd_fix_can_bus']['speed'].view(-1, 1)  # [batch, 1]
    
    # Forward pass
    e_p, resnet_inter = model.encoder_embedding_perception(x)
    if return_backbone_features:
        return resnet_inter
    
    e_p = rearrange(e_p, '(B cam) dim h w -> B (cam h w) dim', cam=3)  # [B, S*cam*h*w, D]
    e_d = model.command(d).unsqueeze(1)
    e_s = model.speed(s).unsqueeze(1)
    
    if model.num_register_tokens > 0:
        e_p = torch.cat([model.register_tokens.repeat(batch_size, 1, 1), e_p], dim=1)
    
    e_p = e_p + e_d + e_s
    e_p = e_p + model.positional_encoding
    
    _, attn_weights = model.tx_encoder(e_p)
    
    if return_attentions:
        return attn_weights

def model_forward(
    model: CIL_multiview, 
    data: dict
) -> torch.Tensor:
    # Get the batch size
    batch_size = data['ar_resized_rgb_central'].shape[0]
    cam = len([c for c in g_conf.DATA_USED if 'rgb' in c])  # Number of cameras

    # Stack RGB images [batch, 3, h, w]
    src_images = [data[camera_type] for camera_type in g_conf.DATA_USED if 'rgb' in camera_type]
    x = torch.stack(src_images, dim=1)  # [batch, cam_num, 3, h, w]
    x = rearrange(x, 'B cam C H W -> (B cam) C H W')
    
    # Process directions
    d = data['cmd_fix_can_bus']['direction'].float()  # [batch, 4]
    # d = torch.tensor(directions, device='cuda').float().view(-1, 1)  # [batch, 1]
    
    # Process speeds
    s = data['cmd_fix_can_bus']['speed'].view(-1, 1)  # [batch, 1]
    
    # Forward pass
    e_p, _ = model.encoder_embedding_perception(x)
    
    e_p = rearrange(e_p, '(B cam) dim h w -> B (cam h w) dim', cam=cam)  # [B, S*cam*h*w, D]
    e_d = model.command(d).unsqueeze(1)
    e_s = model.speed(s).unsqueeze(1)
    
    if model.num_register_tokens > 0:
        e_p = torch.cat([model.register_tokens.repeat(batch_size, 1, 1), e_p], dim=1)
    
    e_p = e_p + e_d + e_s
    e_p = e_p + model.positional_encoding
    
    in_memory, _ = model.tx_encoder(e_p)

    # Get the action output (if no decoder is used, the sa and mha weights are None)
    action_output, _, _ = model.action_prediction(in_memory, cam)  # [B, 1, t=len(TARGETS)]

    return action_output  # [B, 1, len(TARGETS)]


def open_image(path: str) -> Image.Image:
    """Safe image loading with error handling"""
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to load image {path}: {str(e)}")


def get_data_paths(
    base_path: str,
    user_num: int,
    town: str,
    route: str,
    world_tick: int,
    weather: str = "ClearNoon"
) -> Dict[str, str]:
    """Generate paths for synchronized data."""
    rgb_base = Path(base_path) / f"D{user_num:03d}" / town / route / "RGB"
    
    paths = {
        'ar_resized_rgb_central': str(rgb_base / "Central" / weather / 
                          f"D{user_num:03d}_{town}_{route}_RGB_Central_{weather}_{world_tick:07d}.jpg"),
        'ar_resized_rgb_left': str(rgb_base / "Left" / weather / 
                       f"D{user_num:03d}_{town}_{route}_RGB_Left_{weather}_{world_tick:07d}.jpg"),
        'ar_resized_rgb_right': str(rgb_base / "Right" / weather / 
                        f"D{user_num:03d}_{town}_{route}_RGB_Right_{weather}_{world_tick:07d}.jpg"),
        'cmd_fix_can_bus': str(Path(base_path) / f"D{user_num:03d}" / town / route / "CB" / 
                      f"D{user_num:03d}_{town}_{route}_CB_{world_tick:07d}.json")
    }
    return paths


# # === New; time: 10 min, 1.06s/item === 5 to 6 times faster
# # Increasing batch size 32 -> 256; time: 6.5 min, 0.68s/"item", where item is bs=32

def load_single_datapoint(
    base_path: str,
    user_num: int, 
    town: str,
    route: str,
    world_tick: int,
    weather: str,
    image_shape: tuple
) -> dict:
    paths = get_data_paths(base_path, user_num, town, route, world_tick, weather)
    data_point = {}
    
    # Load RGB images
    for cam_type in ['ar_resized_rgb_central', 'ar_resized_rgb_left', 'ar_resized_rgb_right']:
        data_point[cam_type] = open_image(paths[cam_type])
    
    # Load CAN bus data
    with open(paths['cmd_fix_can_bus'], 'r') as f:
        canbus_data = json.loads(f.read())
        data_point['cmd_fix_can_bus'] = canbus_normalization(
            canbus_data, 
            g_conf.DATA_NORMALIZATION,
            ted_normalization=True
        )
    
    # Transform data point
    return ted_transform(data_point, image_shape)

def process_futures(futures: list, batch_size: int) -> Dict[str, torch.Tensor]:
    batch_data = {
        'ar_resized_rgb_central': [],
        'ar_resized_rgb_left': [],
        'ar_resized_rgb_right': [],
        'cmd_fix_can_bus': {'direction': [], 'speed': []}
    }
    
    for future in futures:
        transformed_point = future.result()
        for k, v in transformed_point.items():
            if k != 'cmd_fix_can_bus':
                batch_data[k].append(v)
            else:
                batch_data[k]['direction'].append(v['direction'])
                batch_data[k]['speed'].append(v['speed'])
    
    # Stack tensors
    final_batch = {}
    for k, v in batch_data.items():
        if k != 'cmd_fix_can_bus':
            final_batch[k] = torch.stack(v)
        else:
            final_batch[k] = {
                'direction': torch.tensor(v['direction']),
                'speed': torch.tensor(v['speed'])
            }
    
    return final_batch

def load_data_batch(
    base_path: str,
    user_num: int,
    town: str,
    route: str,
    world_ticks: List[int],
    batch_size: int = 32,
    weather: str = "ClearNoon",
    num_workers: int = 8
) -> Generator[Tuple[Dict[str, torch.Tensor], List[int]], None, None]:
    tick_batches = [world_ticks[i:i + batch_size] 
                   for i in range(0, len(world_ticks), batch_size)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_ticks in tqdm(tick_batches, desc="Processing batches", dynamic_ncols=True):
            futures = []
            for world_tick in batch_ticks:
                futures.append(
                    executor.submit(
                        load_single_datapoint,
                        base_path,
                        user_num,
                        town,
                        route, 
                        world_tick,
                        weather,
                        tuple(g_conf.IMAGE_SHAPE)
                    )
                )
            
            yield process_futures(futures, batch_size), batch_ticks

# ====

def batch_forward(
    model: CIL_multiview,
    base_path: str,
    save_path: str,
    user_num: int,
    town: str,
    route: str,
    world_ticks: List[int],
    attention_name: str,
    batch_size: int = 32
):
    """Process data through model and get the output actions."""
    if save_path is None:
        # Default save path
        save_path = str(Path(base_path) / f"D{user_num:03d}" / town / route / "AO" / f"{attention_name}")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        data_loader = load_data_batch(
            base_path, user_num, town, route, world_ticks, batch_size
        )
        
        for batch_data, batch_ticks in data_loader:
            # Move batch to GPU
            batch_data = {
                k: (
                    {
                        'direction': v['direction'].cuda(),
                        'speed': v['speed'].cuda()
                    } if k == 'cmd_fix_can_bus' else v.cuda()
                ) for k, v in batch_data.items()
            }

            # Get attention weights
            actions = model_forward(model, batch_data)

            for idx, world_tick in enumerate(batch_ticks):
                # Save as .npz
                save_file = save_path / f"D{user_num:03d}_{town}_{route}_{attention_name}_{world_tick:07d}.npz"
                np.savez(save_file, action=actions[idx].cpu().numpy())


def process_and_save_attention(
    model: CIL_multiview,
    base_path: str,
    save_path: str,
    user_num: int,
    town: str,
    route: str,
    world_ticks: List[int],
    attention_name: str,
    batch_size: int = 32,
    att_loss: bool = False,
    att_rollout: bool = False,
    att_affinity: bool = False
):
    """Process data through model and save attention maps."""
    if save_path is None:
        # Default save path
        save_path = str(Path(base_path) / f"D{user_num:03d}" / town / route / "AM" / f"{attention_name}")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()

    
    with torch.no_grad():
        data_loader = load_data_batch(
            base_path, user_num, town, route, world_ticks, batch_size
        )
        
        for batch_data, batch_ticks in data_loader:
            # Move batch to GPU
            batch_data = {
                k: (
                    {
                        'direction': v['direction'].cuda(),
                        'speed': v['speed'].cuda()
                    } if k == 'cmd_fix_can_bus' else v.cuda()
                ) for k, v in batch_data.items()
            }

            # Get attention weights
            attn_weights = model_forward_get_features(model, batch_data, return_attentions=True)  # list of L tensors of shape [B, N, N]

            if att_rollout:
                attn_weights = utils.attn_rollout(attn_weights)

            elif att_affinity:
                # Element-wise multiplication from the last layer to the first
                for idx, world_tick in enumerate(batch_ticks):
                    # Start with the last layer
                    attn = attn_weights[-1]  
                    num_register_tokens = g_conf.NUM_REGISTER_TOKENS
                    attn = attn[idx, num_register_tokens:, num_register_tokens:] # [N, N]
                    for layer in reversed(attn_weights[:-1]):
                        attn = attn * layer[idx, num_register_tokens:, num_register_tokens:]

                    # Average across heads                    
                    attn = attn.mean(dim=0).squeeze()

                    # Reshape to [h, w] format
                    reshaped_attn = rearrange(
                        attn,
                        '(h w cam) -> h (w cam)' if att_loss else '(cam h w) -> h (cam w)',
                        cam=3, h=model.res_out_h
                    )
                    
                    # Save as .npz
                    save_file = save_path / f"D{user_num:03d}_{town}_{route}_{attention_name}_{world_tick:07d}.npz"
                    np.savez(save_file, attention=reshaped_attn.cpu().numpy())
            
            else:
                # Vanilla case
                for idx, world_tick in enumerate(batch_ticks):
                    # Process attention weights as shown in plot_attention_weights_on_images
                    attn = attn_weights[-1]  # Last layer
                    # Average across heads and remove register tokens if present
                    num_register_tokens = g_conf.NUM_REGISTER_TOKENS
                    averaged_attn = attn[idx, num_register_tokens:, num_register_tokens:].mean(dim=0).squeeze()
                    
                    # Reshape to [h, w] format
                    reshaped_attn = rearrange(
                        averaged_attn,
                        '(h w cam) -> h (w cam)' if att_loss else '(cam h w) -> h (cam w)',
                        cam=3, h=model.res_out_h
                    )
                    
                    # Save as .npz
                    save_file = save_path / f"D{user_num:03d}_{town}_{route}_{attention_name}_{world_tick:07d}.npz"
                    np.savez(save_file, attention=reshaped_attn.cpu().numpy())


def find_synchronized_ticks(
    base_path: str,
    user_num: int,
    town: str = "Town01",
    route: str = "R01",
    weather: str = "ClearNoon"
) -> List[int]:
    """Find ticks where we have all required data (RGB + CAN bus)."""
    rgb_base = Path(base_path) / f"D{user_num:03d}" / town / route / "RGB"
    cb_base = Path(base_path) / f"D{user_num:03d}" / town / route / "CB"
    
    # Get ticks from each source
    ticks = {
        'ar_resized_rgb_central': set(),
        'ar_resized_rgb_left': set(),
        'ar_resized_rgb_right': set(),
        'cmd_fix_can_bus': set()
    }
    
    # Get CAN bus ticks
    for cb_file in tqdm(list(cb_base.glob(f"D{user_num:03d}_{town}_{route}_CB_*.json")), 
                       desc="Scanning CAN bus files", dynamic_ncols=True):
        tick = int(cb_file.stem.split('_')[-1])
        ticks['cmd_fix_can_bus'].add(tick)
    
    # Get RGB ticks for each camera
    for camera in ['Central', 'Left', 'Right']:
        rgb_path = rgb_base / camera / weather
        for rgb_file in tqdm(list(rgb_path.glob(f"D{user_num:03d}_{town}_{route}_RGB_{camera}_{weather}_*.jpg")),
                            desc=f"Scanning {camera} camera", dynamic_ncols=True):
            tick = int(rgb_file.stem.split('_')[-1])
            ticks[f'ar_resized_rgb_{camera.lower()}'].add(tick)
    
    # Find common ticks
    synchronized_ticks = set.intersection(*ticks.values())
    
    print(f"\nFound {len(synchronized_ticks)} synchronized ticks")
    for source, source_ticks in ticks.items():
        print(f"{source}: {len(source_ticks)} ticks")
    
    return sorted(list(synchronized_ticks))


def load_attention(npz_path: str) -> np.ndarray:
    """
    Load the attention map from a .npz file.

    Args:
        npz_path (str): Path to the .npz file.

    Returns:
        np.ndarray: The attention map.
    """
    return np.load(npz_path)['attention']


def resize_attention(attention: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize the attention map to the new size.

    Args:
        attention (np.ndarray): The attention map of shape (H, W).
        new_size (Tuple[int, int]): The new size (height, width).

    Returns:
        np.ndarray: The resized attention map.
    """
    return cv2.resize(attention, new_size, interpolation=cv2.INTER_LINEAR)


def get_attention_img_upscaled(npz_path: str, new_size: Tuple[int, int]) -> np.ndarray:
    """
    Load and resize the attention map to the new size.

    Args:
        npz_path (str): Path to the .npz file.
        new_size (Tuple[int, int]): The new size (height, width).

    Returns:
        np.ndarray: The resized attention map.
    """
    attention = load_attention(npz_path)
    # Normalize the attention map; [min, max] -> [0, 1]
    attention = utils.min_max_norm(attention)
    # [0, 1] -> [0, 255]
    attention = (attention * 255).astype(np.uint8)
    # If 3D, it's of shape 1 x H x W, so we remove the first dimension
    if len(attention.shape) == 3 and attention.shape[0] == 1:
        attention = attention[0]
    return resize_attention(attention, new_size)


def save_attention_img(npz_path: str, new_size: Tuple[int, int], save_path: str) -> None:
    """
    Load and resize the attention map to the new size, and save it as an image.

    Args:
        npz_path (str): Path to the .npz file.
        new_size (Tuple[int, int]): The new size (height, width).
        save_path (str): Path to save the image.
    """
    attention_img = get_attention_img_upscaled(npz_path, new_size)
    cv2.imwrite(save_path, attention_img)


def load_frame_data(
    world_tick: int,
    paths: Dict[str, str],
    frame_buffer: np.ndarray,
    attention_cache: Dict[int, np.ndarray]
) -> Tuple[dict, np.ndarray]:
    """Load single frame data efficiently using pre-allocated buffer"""
    images = {k: cv2.imread(v) for k, v in paths.items() if 'rgb' in k}
    with open(paths['can_bus']) as f:
        can_data = json.load(f)
    
    # Use cached attention if available
    if world_tick in attention_cache:
        attention_colored = attention_cache[world_tick]
    else:
        attention = get_attention_img_upscaled(paths['attention'], (frame_buffer.shape[1], frame_buffer.shape[0]//2))
        attention_colored = cv2.applyColorMap(attention, cv2.COLORMAP_JET)
        attention_cache[world_tick] = attention_colored
    
    return can_data, images, attention_colored


def process_frame(
    data: Tuple[dict, dict, np.ndarray],
    world_tick: int, 
    frame_buffer: np.ndarray,
    font_settings: dict
) -> np.ndarray:
    """Process single frame using pre-allocated buffer"""
    can_data, images, attention_colored = data
    h, w = images['rgb_central'].shape[:2]

    # Copy images to buffer (faster than creating new array)
    frame_buffer[:h, :w] = images['rgb_left']
    frame_buffer[:h, w:2*w] = images['rgb_central']
    frame_buffer[:h, 2*w:] = images['rgb_right']
    
    # Bottom row with attention overlay
    bottom = cv2.hconcat([images['rgb_left'], images['rgb_central'], images['rgb_right']])
    frame_buffer[h:] = cv2.addWeighted(bottom, 0.7, attention_colored, 0.3, 0)
    
    # Add text overlays
    fs = font_settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Left camera: Frame number and steering
    cv2.putText(frame_buffer, f'Frame: {world_tick:07d}', 
                (fs['x_margin'], fs['y_top']), 
                font, fs['font_scale'], (0, 0, 255), fs['thickness'])
    cv2.putText(frame_buffer, f'Steering: {can_data["steer"]:.2f}', 
                (fs['x_margin'], fs['y_bottom']), 
                font, fs['font_scale'], (0, 255, 255), fs['thickness'])
    
    # Center camera: Command, tag and position
    cv2.putText(frame_buffer, f"{decode_float_directions_to_str(can_data['direction'])}", 
                (w + fs['x_margin'], fs['y_top']), 
                font, fs['font_scale'], (255, 0, 0), fs['thickness'])
    cv2.putText(frame_buffer, f"{can_data['tag']}", 
                (w + fs['x_margin'], fs['y_command']), 
                font, fs['font_scale']/2, (255, 0, 0), fs['thickness'])
    
    pos = can_data['ego_position']
    cv2.putText(frame_buffer, f'Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})', 
                (w + fs['x_margin'], fs['y_bottom']), 
                font, fs['font_scale']/2, (0, 255, 255), fs['thickness'])
    
    # Right camera: Speed and acceleration
    cv2.putText(frame_buffer, f'Speed: {can_data["speed"]:.1f} km/h', 
                (2*w + fs['x_margin'], fs['y_top']), 
                font, fs['font_scale'], (255, 0, 0), fs['thickness'])
    cv2.putText(frame_buffer, f'Acceleration: {can_data["acceleration"]:.2f}', 
                (2*w + fs['x_margin'], fs['y_bottom']), 
                font, fs['font_scale'], (0, 255, 255), fs['thickness'])
    
    return frame_buffer.copy()


def create_attention_visualization_video(
    base_path: str,
    save_path: str,
    output_path: str,
    user_num: int,
    town: str,
    route: str,
    attention_name: str,
    world_ticks: List[int],
    fps: int = 25,
    weather: str = "ClearNoon",
    num_workers: int = 4,
    buffer_size: int = 32
):
    # Initialize video writer and buffers
    rgb_path = Path(base_path) / f"D{user_num:03d}" / town / route / "RGB" / "Central" / weather
    first_img = cv2.imread(str(rgb_path / f"D{user_num:03d}_{town}_{route}_RGB_Central_{weather}_{world_ticks[0]:07d}.jpg"))
    h, w = first_img.shape[:2]
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    video_path = output_path / f"D{user_num:03d}_{town}_{route}_{attention_name}.mp4"
    
    video = cv2.VideoWriter(str(video_path), 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (3*w, 2*h))
    
    frame_buffer = np.zeros((2*h, 3*w, 3), dtype=np.uint8)
    attention_cache = {}
    
    font_settings = {
        'font_scale': w / 300.0,
        'thickness': max(1, int(2 * w / 300.0)),
        'x_margin': int(10 * w / 300.0),
        'y_top': int(30 * w / 300.0),
        'y_command': int(60 * w / 300.0),
        'y_bottom': h - int(30 * w / 300.0)
    }
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Main progress bar for batch processing
        with tqdm(total=len(world_ticks), desc="Processing frames", unit="frame", dynamic_ncols=True) as pbar:
            for i in range(0, len(world_ticks), buffer_size):
                batch_ticks = world_ticks[i:i + buffer_size]
                futures = []
                
                # Submit batch of frames for processing
                for world_tick in batch_ticks:
                    paths = {
                        'rgb_central': str(rgb_path.parent.parent / "Central" / weather / 
                                    f"D{user_num:03d}_{town}_{route}_RGB_Central_{weather}_{world_tick:07d}.jpg"),
                        'rgb_left': str(rgb_path.parent.parent / "Left" / weather / 
                                    f"D{user_num:03d}_{town}_{route}_RGB_Left_{weather}_{world_tick:07d}.jpg"),
                        'rgb_right': str(rgb_path.parent.parent / "Right" / weather / 
                                    f"D{user_num:03d}_{town}_{route}_RGB_Right_{weather}_{world_tick:07d}.jpg"),
                        'can_bus': str(Path(base_path) / f"D{user_num:03d}" / town / route / "CB" / 
                                    f"D{user_num:03d}_{town}_{route}_CB_{world_tick:07d}.json"),
                        'attention': str(Path(save_path) / f"D{user_num:03d}_{town}_{route}_{attention_name}_{world_tick:07d}.npz")
                    }
                    
                    future = executor.submit(load_frame_data, 
                                          world_tick, 
                                          paths, 
                                          frame_buffer,
                                          attention_cache)
                    futures.append((world_tick, future))
                
                # Process and write frames as they complete
                for world_tick, future in futures:
                    frame_data = future.result()
                    frame = process_frame(frame_data, world_tick, frame_buffer.copy(), font_settings)
                    video.write(frame)
                    pbar.update(1)
                
                if len(attention_cache) > buffer_size * 2:
                    attention_cache.clear()
    
    video.release()


def find_available_user_routes(base_path: str) -> Dict[int, List[Tuple[str, str]]]:
    """Find all available town/route combinations for each user in the dataset.
    
    Args:
        base_path: Path to the base directory containing user data
        
    Returns:
        Dictionary mapping user numbers to lists of (town, route) tuples
    """
    available_routes = {}
    base_path = Path(base_path)
    
    # Find all user directories (format: D001, D002, etc.)
    user_dirs = list(base_path.glob("D[0-9][0-9][0-9]"))
    
    for user_dir in user_dirs:
        user_num = int(user_dir.name[1:])  # Convert D019 -> 19
        available_routes[user_num] = []
        
        # Check each town directory
        for town_dir in user_dir.glob("Town*"):
            town = town_dir.name
            # Find all route directories
            for route_dir in town_dir.glob("R*"):
                route = route_dir.name
                # Verify that both RGB and CB directories exist
                if (route_dir / "RGB").exists() and (route_dir / "CB").exists():
                    available_routes[user_num].append((town, route))
    
    return available_routes

def validate_user_route_combinations(
    requested_users: List[int],
    town: str,
    route: str,
    available_routes: Dict[int, List[Tuple[str, str]]]
) -> List[int]:
    """Validate and filter user list based on route availability.
    
    Args:
        requested_users: List of user numbers requested for processing
        town: Requested town
        route: Requested route
        available_routes: Dictionary of available routes per user
        
    Returns:
        List of valid users that have the requested route
    """
    valid_users = []
    unavailable_users = []
    
    for user in requested_users:
        if user not in available_routes:
            unavailable_users.append(f"User {user} not found in dataset")
            continue
            
        if (town, route) not in available_routes[user]:
            available = ", ".join([f"{t}/{r}" for t, r in available_routes[user]])
            unavailable_users.append(
                f"User {user} does not have route {town}/{route} "
                f"(available: {available})"
            )
            continue
            
        valid_users.append(user)
    
    # Print warnings for unavailable users
    if unavailable_users:
        click.secho("\nWarnings:", fg='yellow')
        for warning in unavailable_users:
            click.secho(f"- {warning}", fg='yellow')
    
    if not valid_users:
        raise click.BadParameter(
            f"None of the requested users have route {town}/{route} available"
        )
    
    return valid_users

def parse_user_list(ctx, param, value):
    """Parse comma-separated list of user numbers and ranges.
    
    Examples:
        "1,2,3" -> [1, 2, 3]
        "1-5" -> [1, 2, 3, 4, 5]
        "1-3,7,9-11" -> [1, 2, 3, 7, 9, 10, 11]
    """
    try:
        result = set()
        # Split by comma and process each part
        for part in value.split(','):
            part = part.strip()
            if '-' in part:
                # Handle range
                start, end = map(int, part.split('-'))
                if start > end:
                    raise click.BadParameter(f'Invalid range: {start} is greater than {end}')
                result.update(range(start, end + 1))
            else:
                # Handle single number
                result.add(int(part))
        return sorted(list(result))
    except ValueError:
        raise click.BadParameter('User input must be comma-separated integers or ranges (e.g., "1-5,7,9-11")')


def process_single_canbus_file(file_path: Path) -> dict:
    """Process a single CAN bus file.
    
    The file path is expected to be in the format:
    .../D{user}/Town{XX}/R{XX}/CB/D{user}_{town}_{route}_CB_{tick}.json
    """
    # Extract metadata from filename
    parts = file_path.stem.split('_')
    user_num = int(parts[0][1:])  # Extract number from D{XXX}
    town = parts[1]
    route = parts[2]
    tick = int(parts[-1])
    
    # Read JSON file
    with open(file_path) as f:
        data = json.load(f)
    
    # Flatten nested arrays and add metadata
    return {
        'tick': tick,
        'user': user_num,
        'town': town,
        'route': route,
        'is_dynamic_setup': SETUP_MAPPING.get(user_num, None),
        'acceleration': data['acceleration'],
        'brake': data['brake'],
        'direction': data['direction'],
        'speed': data['speed'],
        'calc_speed': data['calc_speed'],
        'steer': data['steer'],
        'throttle': data['throttle'],
        'hand_brake': data['hand_brake'],
        'rear_gear': data['rear_gear'],
        'blinker': data['blinker'],
        'tag': data['tag'],
        'pos_x': data['ego_position'][0],
        'pos_y': data['ego_position'][1],
        'pos_z': data['ego_position'][2],
        'gyro_x': data['gyroscope'][0],
        'gyro_y': data['gyroscope'][1],
        'gyro_z': data['gyroscope'][2],
        'accel_x': data['accelerometer'][0],
        'accel_y': data['accelerometer'][1],
        'accel_z': data['accelerometer'][2]
    }


def process_canbus_files_parallel(base_path: str, user_num: int, num_workers: int = None) -> pd.DataFrame:
    """Process all CAN bus files for a specific user across all towns and routes.
    
    Args:
        base_path: Base path containing the data
        user_num: User number to process
        num_workers: Number of worker processes. If None, uses cpu_count()
    """
    user_path = Path(base_path) / f"D{user_num:03d}"
    
    # Find all CAN bus files for this user recursively
    files = list(user_path.rglob("CB/*.json"))
    
    if not files:
        click.secho(f"Warning: No CAN bus files found for user {user_num}", fg='yellow')
        return pd.DataFrame()
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(32, os.cpu_count() + 4)
    
    # Process files using ThreadPoolExecutor
    data_list = []
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and create future->file mapping
        future_to_file = {
            executor.submit(process_single_canbus_file, file_path): file_path 
            for file_path in files
        }
        
        # Process completed futures with progress bar
        for future in tqdm(
            as_completed(future_to_file), 
            total=len(files),
            desc=f"Processing CAN bus files for User {user_num}",
            dynamic_ncols=True
        ):
            file_path = future_to_file[future]
            try:
                data = future.result()
                data_list.append(data)
            except Exception as e:
                failed_files.append((file_path, str(e)))
    
    # Report any failures
    if failed_files:
        click.secho(f"\nWarning: Failed to process {len(failed_files)} files:", fg='yellow')
        for file_path, error in failed_files[:5]:
            click.secho(f"- {file_path}: {error}", fg='yellow')
        if len(failed_files) > 5:
            click.secho(f"  ... and {len(failed_files) - 5} more", fg='yellow')
    
    # Convert to DataFrame
    return pd.DataFrame(data_list)
    


def save_dataset(df: pd.DataFrame, save_path: Path, format: str = 'csv') -> None:
    """Save the processed dataset with date suffix."""
    # Create date suffix
    date_suffix = datetime.now().strftime('%Y%m%d')
    
    # Create filename
    filename = f"ted_dataset_canbus_{date_suffix}"
    
    # Create save directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save according to format
    if format.lower() == 'csv':
        df.to_csv(save_path / f"{filename}.csv", index=False)
    else:  # pkl
        df.to_pickle(save_path / f"{filename}.pkl")


def create_setup_mapping(dynamic_users: list, user_range: str = "1-44") -> dict:
    """Create mapping of users to simulator type (dynamic vs static).
    
    Args:
        dynamic_users: List of user numbers with dynamic simulator
        user_range: String range of all users to consider (e.g. "1-44")
        
    Returns:
        Dictionary mapping user numbers to boolean (True for dynamic, False for static)
    
    Example:
        >>> dynamic_users = [19, 6, 10]
        >>> create_setup_mapping(dynamic_users, "1-5,6,10,19")
        {1: False, 2: False, 3: False, 4: False, 5: False, 6: True, 10: True, 19: True}
    """
    # Convert dynamic users to set for O(1) lookup
    dynamic_set = set(dynamic_users)
    
    # Get all users to consider using parse_user_list
    all_users = parse_user_list(None, None, user_range)
    
    # Create mapping - True if in dynamic_set, False otherwise
    return {user: user in dynamic_set for user in all_users}


dynamic_simulator_users = [19, 6, 10, 23, 22, 41, 7, 42, 34, 16, 40, 20, 29, 4, 12, 13, 9, 36, 26, 27, 21, 43, 5]
SETUP_MAPPING = create_setup_mapping(dynamic_simulator_users, "1-44")


class AttentionMapAnalyzer:
    def __init__(
        self,
        data_path: str,
        canbus_csv: str,
        attention_type: str = 'human',  # 'human' or 'model'
        attention_name: Optional[str] = None,
        output_path: Optional[str] = None,
        num_workers: int = 8
    ):
        """
        Initialize the analyzer with paths and configuration.
        
        Args:
            data_path: Base path containing attention map data
            canbus_csv: Path to the CAN bus CSV file
            attention_type: Either 'human' for human attention/gaze maps or 'model' for model predictions
            attention_name: Name of the model attention (required if attention_type='model')
            output_path: Path for saving outputs (defaults to data_path/analysis)
            num_workers: Number of worker threads for parallel processing
        """
        if attention_type == 'model' and not attention_name:
            raise ValueError("attention_name must be provided when attention_type is 'model'")
            
        self.data_path = Path(data_path)
        self.canbus_df = pd.read_csv(canbus_csv)
        self.attention_type = attention_type
        self.attention_name = attention_name
        if not output_path:
            raise ValueError("output_path must be provided")
        self.output_path = Path(output_path)
        if not self.output_path.exists():
            raise ValueError(f"Output directory {output_path} does not exist")
        if not os.access(self.output_path, os.W_OK):
            raise ValueError(f"No write permission for output directory {output_path}")
        self.num_workers = num_workers
        
    def filter_data(
        self,
        town: Optional[str] = None,
        route: Optional[str] = None,
        user: Optional[int] = None,
        is_dynamic: Optional[bool] = None,
        direction: Optional[int] = None,
        tag: Optional[str] = None,
        min_speed: Optional[float] = None,
        max_speed: Optional[float] = None,
        blinker: Optional[int] = None,  # 0: inactive, 1: right, -1: left
        min_acceleration_g: Optional[float] = None,
        custom_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter CAN bus data based on various criteria.
        """
        df = self.canbus_df.copy()
        
        # Apply filters
        if town:
            df = df[df['town'] == town]
        if route:
            df = df[df['route'] == route]
        if user:
            df = df[df['user'] == user]
        if is_dynamic is not None:
            df = df[df['is_dynamic_setup'] == is_dynamic]
        if direction:
            df = df[df['direction'] == direction]
        if tag:
            df = df[df['tag'] == tag]
        if min_speed is not None:
            df = df[df['speed'] >= min_speed]
        if max_speed is not None:
            df = df[df['speed'] <= max_speed]
        if blinker is not None:
            df = df[df['blinker'] == blinker]
        if min_acceleration_g:
            total_accel = np.sqrt(
                df['accel_x']**2 + 
                df['accel_y']**2 + 
                df['accel_z']**2
            ) / 9.81  # Convert to G forces
            df = df[total_accel >= min_acceleration_g]
        if custom_filter:
            df = df.query(custom_filter)
            
        return df
    
    def generate_output_name(self, filter_params: dict) -> str:
        """Generate a descriptive output name based on filter parameters."""
        parts = []
        
        # Add attention type and model name if applicable
        parts.append('human' if self.attention_type == 'human' else f'model_{self.attention_name}')
        
        # Add main identifiers
        if filter_params.get('town'):
            parts.append(f"{filter_params['town']}")
        if filter_params.get('route'):
            parts.append(f"{filter_params['route']}")
        if filter_params.get('user'):
            parts.append(f"user{filter_params['user']}")
        
        # Add other significant filters
        if filter_params.get('direction'):
            parts.append(f"dir{filter_params['direction']}")
        if filter_params.get('tag'):
            parts.append(f"tag-{filter_params['tag']}")
        if filter_params.get('is_dynamic') is not None:
            parts.append("dynamic" if filter_params['is_dynamic'] else "static")
        if filter_params.get('min_acceleration_g'):
            parts.append(f"min{filter_params['min_acceleration_g']}G")
        if filter_params.get('blinker') is not None:
            blinker_state = {0: 'no_blinker', 1: 'right_blinker', -1: 'left_blinker'}
            parts.append(blinker_state[filter_params['blinker']])
            
        # Add date and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Combine all parts
        return f"{'_'.join(parts)}_{timestamp}"
    
    def load_attention_map(self, file_path: str) -> Optional[np.ndarray]:
        """Load a single attention map file."""
        try:
            # Try to load the file directly without checking existence
            with np.load(file_path) as data:
                attention = data['attention']
                # Verify we got valid data
                if not isinstance(attention, np.ndarray):
                    print(f"Warning: {file_path} contains invalid data type: {type(attention)}")
                    return None
                if attention.size == 0:
                    print(f"Warning: {file_path} contains empty array")
                    return None
                return attention
        except (PermissionError, FileNotFoundError) as e:
            # Log the first few permission/not found errors
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            if self._error_count < 5:
                print(f"Access error for {file_path}: {str(e)}")
                self._error_count += 1
            elif self._error_count == 5:
                print("... suppressing further access errors")
                self._error_count += 1
            return None
        except Exception as e:
            # Log other unexpected errors
            print(f"Unexpected error loading {file_path}: {str(e)}")
            return None
            
    def get_attention_path(
        self,
        user: int,
        town: str,
        route: str,
        tick: int,
        camera: Optional[str] = None
    ) -> str:
        """Generate the full path for an attention map file."""
        if self.attention_type == 'human':
            if not camera:
                raise ValueError("camera must be provided for human attention maps")
            return str(self.data_path / f"D{user:03d}" / town / route / "AET" / camera /
                      f"D{user:03d}_{town}_{route}_AET_{camera}_{tick:07d}.npz")
        else:  # model
            return str(self.data_path / f"D{user:03d}" / town / route / "AM" / self.attention_name /
                      f"D{user:03d}_{town}_{route}_{self.attention_name}_{tick:07d}.npz")
    
    def process_filtered_data(
        self,
        filtered_df: pd.DataFrame,
        cameras: Optional[List[str]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """
        Process filtered data to compute average attention maps.
        """
        if self.attention_type == 'human':
            if not cameras:
                cameras = ['Left', 'Central', 'Right', 'MLeft', 'MRight']
            attention_sums = {cam: None for cam in cameras}
            counts = {cam: 0 for cam in cameras}
        else:
            attention_sums = {'combined': None}
            counts = {'combined': 0}
        
        # Track stats
        attempted = 0
        successful = 0
        failed_paths = []
        
        print("\nAttempting to load attention maps...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            # Submit all loading tasks
            for _, row in filtered_df.iterrows():
                if self.attention_type == 'human':
                    for camera in cameras:
                        path = self.get_attention_path(
                            row['user'], row['town'], row['route'], row['tick'], camera
                        )
                        attempted += 1
                        futures.append((camera, executor.submit(self.load_attention_map, path), path))
                else:  # model
                    path = self.get_attention_path(
                        row['user'], row['town'], row['route'], row['tick']
                    )
                    attempted += 1
                    futures.append(('combined', executor.submit(self.load_attention_map, path), path))
            
            # Process results as they complete
            for camera, future, path in tqdm(futures, desc="Processing attention maps", dynamic_ncols=True):
                attention_map = future.result()
                if attention_map is not None:
                    successful += 1
                    if attention_sums[camera] is None:
                        attention_sums[camera] = attention_map
                    else:
                        attention_sums[camera] += attention_map
                    counts[camera] += 1
                else:
                    failed_paths.append(path)
        
        print("\nProcessing summary:")
        print(f"Attempted to load {attempted} files")
        print(f"Successfully loaded {successful} files")
        print(f"Failed to load {len(failed_paths)} files")
        
        if failed_paths:
            print("\nFirst few failed paths:")
            for path in failed_paths[:5]:
                print(f"  {path}")
            if len(failed_paths) > 5:
                print(f"  ... and {len(failed_paths) - 5} more")
        
        # Calculate averages
        averages = {}
        for camera in attention_sums.keys():
            if counts[camera] > 0:
                averages[camera] = attention_sums[camera] / counts[camera]
        
        return averages, counts
    
    def save_results(
        self,
        attention_maps: Dict[str, np.ndarray],
        counts: Dict[str, int],
        filter_params: dict
    ) -> str:
        """Save the analysis results."""
        # Generate output name and create save directory path
        output_name = self.generate_output_name(filter_params)
        save_dir = self.output_path / output_name
        
        # Create the output directory
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create output directory {save_dir}: {str(e)}")
        
        # Save attention maps
        for camera, attention_map in attention_maps.items():
            # Save the attention map as a .npz file
            np.savez(
                save_dir / f"average_attention_{camera}.npz",
                attention=attention_map
            )

            # Save the attention map as an image (upscale to 960x540 for model attention)
            new_size = (960, 540) if self.attention_type == 'model' else (attention_map.shape[1], attention_map.shape[0])
            save_attention_img(
                save_dir / f"average_attention_{camera}.npz",
                new_size,
                save_path=str(save_dir / f"average_attention_{camera}.jpg")
            )

         # Save metadata
        metadata = {
            'attention_type': self.attention_type,
            'attention_name': self.attention_name if self.attention_type == 'model' else None,
            'filter_params': filter_params,
            'processing_stats': {
                'total_samples': sum(counts.values()),  # Total number of successfully processed samples
                'samples_per_camera': counts
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return str(save_dir)

    def analyze(self, **filter_params) -> Tuple[Dict[str, np.ndarray], str]:
        """Perform complete analysis pipeline with given filters."""
        # Filter data
        filtered_df = self.filter_data(**filter_params)
        
        if filtered_df.empty:
            raise ValueError("No data matches the specified criteria")
            
        print(f"Found {len(filtered_df)} matching samples")
        
        # Process data
        attention_maps, counts = self.process_filtered_data(filtered_df)
        
        if not attention_maps:
            raise ValueError("No attention maps could be processed")
        
        # Save results
        save_path = self.save_results(attention_maps, counts, filter_params)
        
        return attention_maps, counts, save_path


# ======== CLI COMMANDS ========


def attention_common_options(f):
    options = [
        click.option('--base-path', required=True, help='Base path for data (e.g., /data/users/)'),
        click.option('--save-path', default=None, help='Path to save outputs (attention maps or videos)'),
        click.option('--user-nums', required=True, callback=parse_user_list, 
                    help='Ranges, comma-separated values, or combination of users to analyze (e.g., "19-23,27" results in [19, 20, 21, 22, 23, 27])'),
        click.option('--town', required=True, help='Town name (e.g., Town01)'),
        click.option('--route', required=True, help='Route name (e.g., R01)')
    ]
    for option in reversed(options):
        f = option(f)
    return f


@click.group()
def cli():
    """Tool for attention map prediction and visualization."""
    pass


@cli.command(name='predict-attention')
@attention_common_options
@click.option('--exp-batch', required=True, help='Experiment batch name (folder in configs/)')
@click.option('--exp-name', required=True, help='Experiment name (YAML file name without extension)')
@click.option('--checkpoint-number', default=40, show_default=True, help='Checkpoint number to load')
@click.option('--batch-size', default=256, show_default=True, help='Batch size for processing')
@click.option('--attention-name', required=True, help='Name for attention maps (e.g., "AM-CPP-AttLoss-400x225")')
@click.option('--att-rollout/--no-att-rollout', default=False, show_default=True)
@click.option('--att-affinity/--no-att-affinity', default=False, show_default=True)
@click.option('--gpu', default=0, show_default=True, help='GPU device number to use', type=click.IntRange(min=0))
def predict_attention(base_path, save_path, user_nums, town, route, attention_name,
                     exp_batch, exp_name, checkpoint_number, batch_size,
                     att_rollout, att_affinity, gpu):
    """Predict attention maps for given static dataset."""
    # Set up CUDA and model configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    # Find available routes and validate user selections
    available_routes = find_available_user_routes(base_path)
    valid_users = validate_user_route_combinations(user_nums, town, route, available_routes)
    
    if len(valid_users) < len(user_nums):
        click.secho(f"\nProceeding with {len(valid_users)} out of {len(user_nums)} requested users", 
                   fg='yellow')
    
    # Load model configuration and initialize model
    merge_with_yaml(os.path.join('configs', exp_batch, f'{exp_name}.yaml'))
    g_conf.PROCESS_NAME = 'train_val'
    
    model = CIL_multiview(g_conf.MODEL_CONFIGURATION, average_attn_weights=True).to('cuda')
    checkpoint_path = os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results', 
                               g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME, 'checkpoints')
    model = load_model_from_checkpoint(model, checkpoint_path, checkpoint_number=checkpoint_number)
    
    # Process each valid user
    for user_num in tqdm(valid_users, desc="Processing users", dynamic_ncols=True):
        click.echo(f"\nProcessing user {user_num}")

        # Find synchronized ticks
        world_ticks = find_synchronized_ticks(base_path=base_path, user_num=user_num, 
                                           town=town, route=route)

        # Process data and save attention maps
        process_and_save_attention(
            model=model, base_path=base_path, save_path=save_path,
            user_num=user_num, town=town, route=route,
            world_ticks=world_ticks, attention_name=attention_name,
            batch_size=batch_size, att_loss=g_conf.ATTENTION_LOSS,
            att_rollout=att_rollout, att_affinity=att_affinity
        )


@cli.command(name='visualize-attention')
@attention_common_options
@click.option('--fps', default=25, show_default=True)
@click.option('--weather', default="ClearNoon", show_default=True)
@click.option('--attention-name', required=True, help='Name for attention maps (e.g., "AM-CPP-AttLoss-400x225")')
@click.option('--num-workers', default=4, show_default=True)
@click.option('--buffer-size', default=32, show_default=True)
def create_visualization(base_path, save_path, user_nums, town, route,
                       attention_name, fps, weather, num_workers, buffer_size):
    """Create visualization video from the aforementioned attention maps."""
    # Find available routes and validate user selections    
    available_routes = find_available_user_routes(base_path)
    valid_users = validate_user_route_combinations(user_nums, town, route, available_routes)
    
    if len(valid_users) < len(user_nums):
        click.secho(f"\nProceeding with {len(valid_users)} out of {len(user_nums)} requested users", 
                   fg='yellow')

    for user_num in valid_users:
        click.echo(f"\nProcessing video for user {user_num}")
        world_ticks = find_synchronized_ticks(base_path=base_path, user_num=user_num,
                                           town=town, route=route, weather=weather)

        if save_path is None:
            # Default save path
            save_path = str(Path(base_path) / f"D{user_num:03d}" / town / route / "AM" / f"{attention_name}")

        create_attention_visualization_video(
            base_path=base_path, 
            save_path=save_path, 
            output_path=save_path,
            user_num=user_num, 
            town=town, 
            route=route,
            attention_name=attention_name, 
            world_ticks=world_ticks,
            fps=fps, 
            weather=weather, 
            num_workers=num_workers,
            buffer_size=buffer_size
        )



@cli.command(name='extract-canbus')
@click.option('--base-path', required=True, help='Base path for data (e.g., /data/users/)')
@click.option('--save-path', default=None, help='Path to save outputs')
@click.option('--user-nums', required=True, callback=parse_user_list, 
              help='Comma-separated list of user numbers and ranges (e.g., "1-5,7,9-11")')
@click.option('--format', type=click.Choice(['csv', 'pkl']), default='csv', 
              help='Output file format')
@click.option('--num-workers', default=None, type=int,
              help='Number of worker threads per user. Defaults to min(32, cpu_count + 4)')
def extract_canbus(base_path, save_path, user_nums, format, num_workers):
    """Extract and process CAN bus data for analysis across all towns and routes."""
    # Process data for each valid user
    all_data = []
    for user_num in user_nums:
        click.echo(f"\nProcessing CAN bus data for user {user_num}")
        user_data = process_canbus_files_parallel(base_path, user_num, num_workers)
        if not user_data.empty:
            all_data.append(user_data)
    
    if not all_data:
        click.secho("No data was processed!", fg='red')
        return
    
    # Combine all user data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Set default save path if none provided
    if save_path is None:
        save_path = Path(base_path) / "processed_data"
    else:
        save_path = Path(save_path)
    
    # Save the combined dataset
    save_dataset(combined_data, save_path, format)
    
    # Print summary
    click.echo("\nProcessing Summary:")
    click.echo(f"Total records: {len(combined_data)}")
    click.echo(f"Unique users: {combined_data['user'].nunique()}")
    click.echo(f"Unique towns: {combined_data['town'].nunique()}")
    click.echo(f"Unique routes: {combined_data['route'].nunique()}")
    click.echo(f"\nData successfully saved to {save_path}")


@cli.command(name='plot-data')
@click.option('--data-path', required=True, help='Base path containing attention map data')
@click.option('--canbus-csv', required=True, help='Path to CAN bus CSV file')
@click.option('--attention-type', type=click.Choice(['human', 'model']), required=True,
              help='Type of attention maps to analyze')
@click.option('--attention-name', help='Name of model attention (required if type is "model")')
@click.option('--output-path', required=True, 
              help='Path for saving outputs (directory must exist and be writable)')
@click.option('--town', help='Town name (e.g., Town01)')
@click.option('--route', help='Route name (e.g., R01)')
@click.option('--user', type=int, help='Specific user number')
@click.option('--is-dynamic', type=bool, help='Filter for dynamic/static simulator')
@click.option('--direction', type=int, help='Specific direction (1-4)')
@click.option('--tag', help='Specific scenario tag')
@click.option('--min-speed', type=float, help='Minimum speed in km/h')
@click.option('--max-speed', type=float, help='Maximum speed in km/h')
@click.option('--blinker', type=click.Choice(['0', '1', '-1']), 
              help='Blinker state (0: inactive, 1: right, -1: left)')
@click.option('--min-acceleration-g', type=float, help='Minimum total acceleration in G forces')
@click.option('--custom-filter', help='Custom pandas query string')
@click.option('--num-workers', default=8, help='Number of worker threads')
def main(**kwargs):
    """Analyze attention maps with flexible filtering based on CAN bus data."""
    try:
        # Convert blinker to int if provided
        if kwargs.get('blinker') is not None:
            kwargs['blinker'] = int(kwargs['blinker'])
        
        analyzer = AttentionMapAnalyzer(
            kwargs.pop('data_path'),
            kwargs.pop('canbus_csv'),
            kwargs.pop('attention_type'),
            kwargs.pop('attention_name'),
            kwargs.pop('output_path'),
            kwargs.pop('num_workers')
        )
        
        attention_maps, counts, save_path = analyzer.analyze(**kwargs)
        
        click.echo(f"\nAnalysis comsplete! Results saved to: {save_path}")
        for camera, count in counts.items():
            click.echo(f"Processed {count} samples for {camera} camera")
            
    except ValueError as e:
        click.echo(f"\nError: {str(e)}", err=True)
        exit(1)
    except Exception as e:
        click.echo(f"\nUnexpected error: {str(e)}", err=True)
        exit(1)


if __name__ == '__main__':
    cli()