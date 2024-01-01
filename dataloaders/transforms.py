import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import cv2

from configs import g_conf
import os


def canbus_normalization(can_bus_dict, data_ranges):
    for dtype in data_ranges.keys():
        # we normalize the steering in range of [-1.0, 1.0]
        if dtype in ['steer', 'acceleration']:
            if not data_ranges[dtype] == [-1.0, 1.0]:
                print('')
                print('normalizing data for ', dtype)
                print('')
                # we normalize steering data if they are not from [-1.0, 1.0]
                can_bus_dict[dtype] = 2 * ((can_bus_dict[dtype] - data_ranges[dtype][0]) / (data_ranges[dtype][1] - data_ranges[dtype][0])) - 1   # [-1.0, 1.0]
        elif dtype in ['throttle', 'brake', 'speed']:
            # we normalize the other can bus data in range of [0.0, 1.0]
            if not data_ranges[dtype] == [0.0, 1.0]:
                can_bus_dict[dtype] = abs(can_bus_dict[dtype]-data_ranges[dtype][0])/(data_ranges[dtype][1]-data_ranges[dtype][0])     # [0.0, 1.0]

        else:
            raise KeyError(f'The transformation of this data type has not yet defined: {dtype}')

    if 'direction' in can_bus_dict.keys():
        # we encode directions to one-hot vector
        if g_conf.DATA_COMMAND_ONE_HOT:
            if g_conf.DATA_COMMAND_CLASS_NUM == 4:
                can_bus_dict['direction'] = encode_directions_4(can_bus_dict['direction'])
            elif g_conf.DATA_COMMAND_CLASS_NUM == 6:
                can_bus_dict['direction'] = encode_directions_6(can_bus_dict['direction'])
        else:
            # we remark directions from 1-4 to 0-3 for torch.embedding layer
            can_bus_dict['direction'] = [can_bus_dict['direction']-1]
    return can_bus_dict

def train_transform(data: dict, image_shape: 'tuple[int]', resize_attention: 'tuple[int]' = (10, 10)):
    """
        Apply transformations and augmentations. The
        output is from 0-1 float.
    """
    for camera_type in g_conf.DATA_USED:
        if 'rgb' in camera_type:
            image = data[camera_type]
            image = image.resize((image_shape[2], image_shape[1]))  # Note: Bicubic interpolation by default
            image = transforms.RandAugment(2, 10)(image) if g_conf.RAND_AUGMENT else image
            image - transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)(image) if g_conf.AUG_MIX else image
            image = transforms.ColorJitter(brightness=0.3)(image) if g_conf.COLOR_JITTER else image
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'])
            data[camera_type] = image
        elif 'depth' in camera_type:
            pass
        elif 'ss' in camera_type:
            pass
        elif 'virtual_attention' in camera_type and not g_conf.ATTENTION_AS_INPUT:
            image = data[camera_type]
            image = cv2.resize(np.array(image), resize_attention, interpolation=getattr(cv2, g_conf.VIRTUAL_ATTENTION_INTERPOLATION, cv2.INTER_LINEAR))
            image = TF.to_tensor(image)
            data[camera_type] = image
        elif 'virtual_attention' in camera_type and g_conf.ATTENTION_AS_INPUT:
            image = data[camera_type]
            image = cv2.resize(np.array(image), (image_shape[2], image_shape[1]))
            image = TF.to_tensor(image)
            if g_conf.ATTENTION_AS_NEW_CHANNEL:
                image = TF.normalize(image, [0.5], [0.5])
            data[camera_type] = image
        else:
            raise KeyError(f"The camera type is not defined: {camera_type}")

    return data

def val_transform(data, image_shape, resize_attention: 'tuple[int]' = (10, 10)):
    for camera_type in g_conf.DATA_USED:
        if 'rgb' in camera_type:
            image = data[camera_type]
            ## WE ALREADY PRE-PROCESSED IMAGES TO DESIRED SIZE
            height = image_shape[1]
            width = image_shape[2]
            image = image.resize((width, height))
            image = TF.to_tensor(image)
            image = TF.normalize(image,  mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'])
            data[camera_type] = image
        elif 'depth' in camera_type:
            pass
        elif 'ss' in camera_type:
            pass
        elif 'virtual_attention' in camera_type and not g_conf.ATTENTION_AS_INPUT:
            image = data[camera_type]
            image = cv2.resize(np.array(image), resize_attention, interpolation=getattr(cv2, g_conf.VIRTUAL_ATTENTION_INTERPOLATION, cv2.INTER_LINEAR))
            image = TF.to_tensor(image)
            data[camera_type] = image
        elif 'virtual_attention' in camera_type and g_conf.ATTENTION_AS_INPUT:
            image = data[camera_type]
            image = cv2.resize(np.array(image), (image_shape[2], image_shape[1]))
            image = TF.to_tensor(image)
            if g_conf.ATTENTION_AS_NEW_CHANNEL:
                image = TF.normalize(image, [0.5], [0.5])
            data[camera_type] = image
    return data


def encode_directions_6(directions):
    # TURN_LEFT
    if float(directions) == 1.0:
        return [1, 0, 0, 0, 0, 0]
    # TURN_RIGHT
    elif float(directions) == 2.0:
        return [0, 1, 0, 0, 0, 0]
    # GO_STRAIGHT
    elif float(directions) == 3.0:
        return [0, 0, 1, 0, 0, 0]
    # FOLLOW_LANE
    elif float(directions) == 4.0:
        return [0, 0, 0, 1, 0, 0]
    # CHANGELANE_LEFT
    elif float(directions) == 5.0:
        return [0, 0, 0, 0, 1, 0]
    # CHANGELANE_RIGHT
    elif float(directions) == 6.0:
        return [0, 0, 0, 0, 0, 1]
    else:
        raise ValueError("Unexpcted direction identified %s" % str(directions))

def encode_directions_4(directions):
    # TURN_LEFT
    if float(directions) == 1.0:
        return [1, 0, 0, 0]
    # TURN_RIGHT
    elif float(directions) == 2.0:
        return [0, 1, 0, 0]
    # GO_STRAIGHT
    elif float(directions) == 3.0:
        return [0, 0, 1, 0]
    # FOLLOW_LANE
    elif float(directions) == 4.0:
        return [0, 0, 0, 1]
    else:
        raise ValueError("Unexpcted direction identified %s" % str(directions))

def decode_directions_6(one_hot_direction):
    one_hot_direction = list(one_hot_direction)
    index = one_hot_direction.index(max(one_hot_direction))
    # TURN_LEFT
    if index == 0:
        return 1.0
    # TURN_RIGHT
    elif index == 1:
        return 2.0
    # GO_STRAIGHT
    elif index == 2:
        return 3.0
    # FOLLOW_LANE
    elif index == 3:
        return 4.0
    # CHANGELANE_LEFT
    elif index == 4:
        return 5.0
    # CHANGELANE_RIGHT
    elif index == 5:
        return 6.0
    else:
        raise ValueError("Unexpcted direction identified %s" % one_hot_direction)

def decode_directions_4(one_hot_direction):
    one_hot_direction = list(one_hot_direction)
    index = one_hot_direction.index(max(one_hot_direction))
    # TURN_LEFT
    if index == 0:
        return 1.0
    # TURN_RIGHT
    elif index == 1:
        return 2.0
    # GO_STRAIGHT
    elif index == 2:
        return 3.0
    # FOLLOW_LANE
    elif index == 3:
        return 4.0
    else:
        raise ValueError("Unexpcted direction identified %s" % one_hot_direction)


def inverse_normalize(tensor, mean, std):
    """ Inverse normalization with ImageNet parameters. """
    inv_normalize = transforms.Normalize(
        mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
        std=[1/std[0], 1/std[1], 1/std[2]])

    return inv_normalize(tensor)


# ============= Synthetic Attention Maps =============

SS_CONVERTER = np.uint8([
    0,    # unlabeled
    0,    # building
    0,    # fence
    0,    # other
    1,    # ped
    0,    # pole
    1,    # road line
    0,    # road
    0,    # sidewalk
    0,    # vegetation
    1,    # car
    0,    # wall
    0,    # traffic sign
    0,    # sky
    0,    # ground
    0,    # bridge
    0,    # railtrack
    0,    # guardrail
    1,    # trafficlight
    0,    # static
    0,    # dynamic
    0,    # water
    0,    # terrain
    ])

ss_classes = {
        0: [0, 0, 0],  # None
        1: [70, 70, 70],  # Buildings
        2: [100, 40, 40],  # Fences
        3: [55, 90, 80],  # Other
        4: [220, 20, 60],  # Pedestrians
        5: [153, 153, 153],  # Poles
        6: [157, 234, 50],  # RoadLines
        7: [128, 64, 128],  # Roads
        8: [244, 35, 232],  # Sidewalks
        9: [107, 142, 35],  # Vegetation
        10: [0, 0, 142],  # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0],  # TrafficSigns
        13: [70, 130, 180],  # Sky
        14: [81, 0, 81],  # Ground
        15: [150, 100, 100],  # Bridges
        16: [230, 150, 140],  # RailTracks
        17: [180, 165, 180],  # GuardRails
        18: [250, 170, 30],  # TrafficLights
        19: [110, 190, 160],  # Statics
        20: [170, 120, 50],  # Dynamics
        21: [45, 60, 150],  # Water
        22: [145, 170, 100]  # Terrains
}


def fix_segmentation(segmented_img):
    pass

def read_images(depth_path, segmented_path):
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
    segmented_img = cv2.imread(segmented_path, cv2.IMREAD_UNCHANGED)

    if segmented_img.ndim == 3 and segmented_img.shape[2] == 4:  # Check if it's an RGB image
        segmented_img = segmented_img[:, :, -2:-5:-1]
        segmented_one_channel = np.zeros((segmented_img.shape[0], segmented_img.shape[1]), dtype=np.uint8)
        for label, color in ss_classes.items():
            mask = np.all(segmented_img == np.array(color).reshape(1, 1, 3), axis=2)
            segmented_one_channel[mask] = label
        segmented_img = segmented_one_channel
    return depth_img, segmented_img


def process_depth_image(depth_img):
    depth_not_norm = np.add(depth_img[:,:,0], np.add(np.multiply(depth_img[:,:,1], 256), np.multiply(depth_img[:,:,2], 256**2)))
    processed_depth = np.multiply(np.divide(depth_not_norm, 256**3-1),1000)
    return processed_depth


def create_masks(depth_img, segmentation, converter, road_label: int = 7, sidewalk_label: int = 8,
                 central_camera: bool = False, depth_threshold: float = 20.0, min_depth: float = 2.3):
    min_depth = min_depth if central_camera else 0
    depth_condition = (depth_img < depth_threshold) & (depth_img > min_depth)
    mask_depth = np.where(depth_condition, 255, 0).astype(np.uint8)
    mask_segmentation = np.isin(segmentation, np.where(converter==1)[0]).astype(np.uint8) * 255
    road_mask = (segmentation == road_label).astype(np.uint8)
    sidewalk_mask = (segmentation == sidewalk_label).astype(np.uint8)
    return mask_depth, mask_segmentation, road_mask, sidewalk_mask


def find_boundary(road_mask, sidewalk_mask, dilation_kernel_size: int = 10, dilation_iterations: int = 1):
    dilated_road = cv2.dilate(road_mask, np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8), iterations=dilation_iterations)
    boundary = cv2.bitwise_and(dilated_road, sidewalk_mask)
    boundary = (boundary > 0).astype(np.uint8) * 255
    return boundary


def get_virtual_attention_map(depth_path, segmented_path, central_camera: bool = True, depth_threshold: float = 20.0, min_depth: float = 2.3):
    depth_rgb, segmentation = read_images(depth_path, segmented_path)
    depth_img = process_depth_image(depth_rgb)
    mask_depth, mask_segmentation, road_mask, sidewalk_mask = create_masks(
        depth_img, segmentation, SS_CONVERTER, central_camera=central_camera,
        depth_threshold=depth_threshold, min_depth=min_depth)
    boundary = find_boundary(road_mask, sidewalk_mask)
    merge_boundary = cv2.bitwise_or(boundary, mask_segmentation)
    merge = cv2.bitwise_and(mask_depth, merge_boundary)
    return depth_img, segmentation, mask_depth, mask_segmentation, boundary, merge_boundary, merge
