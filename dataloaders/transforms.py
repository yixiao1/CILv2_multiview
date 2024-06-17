import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import cv2
import PIL

from configs import g_conf


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

def decode_directions_4_str(one_hot_direction):
    one_hot_direction = list(one_hot_direction)
    index = one_hot_direction.index(max(one_hot_direction))
    # TURN_LEFT
    if index == 0:
        return 'TURN LEFT'
    # TURN_RIGHT
    elif index == 1:
        return 'TURN RIGHT'
    # GO_STRAIGHT
    elif index == 2:
        return 'GO STRAIGHT'
    # FOLLOW_LANE
    elif index == 3:
        return 'FOLLOW LANE'
    else:
        raise ValueError("Unexpcted direction identified %s" % one_hot_direction)

def decode_onehot_directions_to_str(one_hot_direction):
    one_hot_direction = list(one_hot_direction)
    index = one_hot_direction.index(max(one_hot_direction))
    # TURN_LEFT
    if index == 0:
        return 'TURN LEFT'
    # TURN_RIGHT
    elif index == 1:
        return 'TURN RIGHT'
    # GO_STRAIGHT
    elif index == 2:
        return 'GO STRAIGHT'
    # FOLLOW_LANE
    elif index == 3:
        return 'FOLLOW LANE'
    # CHANGELANE_LEFT
    elif index == 4:
        return 'CHANGELANE LEFT'
    # CHANGELANE_RIGHT
    elif index == 5:
        return 'CHANGELANE RIGHT'
    else:
        raise ValueError("Unexpcted direction identified %s" % one_hot_direction)


def decode_float_directions_to_str(float_direction):
    if float_direction == 1.0:
        return 'TURN LEFT'
    elif float_direction == 2.0:
        return 'TURN RIGHT'
    elif float_direction == 3.0:
        return 'GO STRAIGHT'
    elif float_direction == 4.0:
        return 'FOLLOW LANE'
    elif float_direction == 5.0:
        return 'CHANGELANE LEFT'
    elif float_direction == 6.0:
        return 'CHANGELANE RIGHT'
    else:
        raise ValueError("Unexpcted direction identified %s" % float_direction)


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
    1,    # Curb
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
        22: [145, 170, 100],  # Terrains
        23: [255, 255, 100]  # Curb; usually from pre-trained models (e.g., Mapillary Vistas)
}


def read_images(image_path, image_type: str = 'depth'):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image_type == 'depth':
        return image
    elif image_type == 'segmentation':
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image = image[:, :, -2:-5:-1]
        image_one_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for label, color in ss_classes.items():
            mask = np.all(image == np.array(color).reshape(1, 1, 3), axis=2)
            image_one_channel[mask] = label
        image = image_one_channel
        return image


def process_depth_image(depth_img):
    if isinstance(depth_img, PIL.Image.Image):
        depth_img = np.array(depth_img)
    depth_not_norm = np.add(depth_img[:, :, 2],
                            np.add(np.multiply(depth_img[:, :, 1], 256), np.multiply(depth_img[:, :, 0], 256 ** 2)))
    processed_depth = np.multiply(np.divide(depth_not_norm, 256 ** 3 - 1), 1000)
    return processed_depth


def create_masks(depth_img, segmentation, converter, road_label: int = 7, sidewalk_label: int = 8, line_label: int = 6,
                 central_camera: bool = False, depth_threshold: float = 20.0, min_depth: float = 2.3):
    mask_segmentation = np.isin(segmentation, np.where(converter == 1)[0]).astype(np.uint8) * 255
    if depth_img is not None:
        min_depth = min_depth if central_camera else 0
        depth_condition = (depth_img < depth_threshold) & (depth_img > min_depth)
        mask_depth = np.where(depth_condition, 255, 0).astype(np.uint8)
    else:
        mask_depth = 255 * np.ones_like(mask_segmentation)
    road_mask = (segmentation == road_label).astype(np.uint8)
    sidewalk_mask = (segmentation == sidewalk_label).astype(np.uint8)
    line_mask = (segmentation == line_label).astype(np.uint8)
    return mask_depth, mask_segmentation, road_mask, sidewalk_mask, line_mask


def find_boundary(road_mask, sidewalk_mask, dilation_kernel_size: int = 10, dilation_iterations: int = 1):
    dilated_road = cv2.dilate(road_mask, np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8),
                              iterations=dilation_iterations)
    boundary = cv2.bitwise_and(dilated_road, sidewalk_mask)
    boundary = (boundary > 0).astype(np.uint8)
    return boundary


def generate_blob_like_perlin_noise(shape, scale=50, layers=3):
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a, b, x):
        return a + x * (b - a)

    def grad(hash, x, y):
        v = hash & 7
        u = x if v < 4 else y
        v = y if v < 4 else x
        return (u if (hash & 1) == 0 else -u) + (v if (hash & 2) == 0 else -v)

    perm = np.random.permutation(256)
    p = np.array([perm[i % 256] for i in range(512)])  # Extend perm array

    def noise(x, y):
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        x -= np.floor(x)
        y -= np.floor(y)
        u = fade(x)
        v = fade(y)
        a = p[X] + Y
        aa = p[a]
        ab = p[a + 1]
        b = p[X + 1] + Y
        ba = p[b]
        bb = p[b + 1]
        return lerp(lerp(grad(p[aa], x, y), grad(p[ba], x - 1, y), u),
                    lerp(grad(p[ab], x, y - 1), grad(p[bb], x - 1, y - 1), u), v)

    total_noise = np.zeros(shape)
    frequency = 1.0
    amplitude = 1.0
    max_value = 0
    for _ in range(layers):
        for i in range(shape[0]):
            for j in range(shape[1]):
                total_noise[i][j] += noise(i / scale * frequency, j / scale * frequency) * amplitude
        max_value += amplitude
        amplitude /= 2
        frequency *= 2

    total_noise = (total_noise + max_value) / (2 * max_value)
    return total_noise


def generate_depth_aware_perlin_noise(depth_not_norm, scale: int = 30, layers: int = 3,
                                      PERMUTATION_SIZE: int = 256):
    depth_not_norm[depth_not_norm > 40] = 40
    # print(depth_not_norm)
    depth_not_norm = 40 - depth_not_norm
    depth_data = (depth_not_norm - np.mean(depth_not_norm) / np.std(depth_not_norm))
    depth_data = np.max(depth_data) - (depth_data - np.min(depth_data)) + 1e-20
    depth_data = depth_data.T

    # cv2.imwrite("depth.png", depth_data)
    # Prepare Perlin noise parameters
    perm = np.random.permutation(PERMUTATION_SIZE)
    p = np.array([perm[i % PERMUTATION_SIZE] for i in range(512)])

    # Define the noise function
    def noise(x, y):
        xi = x.astype(int) & 255
        yi = y.astype(int) & 255
        xf = x - xi
        yf = y - yi
        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)

        n00 = grad(p[p[xi] + yi], xf, yf)
        n01 = grad(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = grad(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = grad(p[p[xi + 1] + yi], xf - 1, yf)

        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)
        return lerp(x1, x2, v)

    def grad(hash, x, y):
        # Convert hash values to 4-bit integers and reshape for broadcasting
        h = (hash & 15).reshape(x.shape)

        # Vectorized computation for u and v
        u = np.where(h < 8, x, y)

        # Handling multiple conditions for v
        v = np.zeros_like(x)
        v_mask = h < 4
        v[v_mask] = y[v_mask]

        # Reshape h to match x and y for broadcasting in the np.where condition
        h_reshaped = h[~v_mask]
        x_reshaped = x[~v_mask]
        v[~v_mask] = np.where((h_reshaped == 12) | (h_reshaped == 14), x_reshaped, 0)

        # Final gradient computation
        result = np.zeros_like(x)
        result += np.where(h & 1 == 0, u, -u)
        result += np.where(h & 2 == 0, v, -v)
        return result

    def lerp(a, b, t):
        return a + t * (b - a)

    # Generate depth-aware Perlin noise
    total_noise = np.zeros(depth_not_norm.shape)
    frequency = 0.2
    amplitude = 10.0

    for _ in range(layers):
        x = np.linspace(0, 1, depth_not_norm.shape[0]) * frequency
        y = np.linspace(0, 1, depth_not_norm.shape[1]) * frequency
        xv, yv = np.meshgrid(x, y, indexing='xy')
        depth_factor = depth_data * scale
        noise_values = noise(xv * depth_factor, yv * depth_factor) * amplitude
        total_noise += noise_values
        frequency *= 2
        amplitude /= 2

    # Normalize the noise
    # total_noise = (total_noise - total_noise.min()) / (total_noise.max() - total_noise.min())
    total_noise = (total_noise + amplitude) / (2 * amplitude)
    return total_noise


def create_mask_noise(width, height, percentage):
    grid_width = width // 128
    grid_height = height // 128

    # Initialize a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Calculate the total and selected number of cells
    total_cells = 128 * 128
    cells_to_apply = int(total_cells * percentage // 100)

    # Randomly select cells
    selected_cells = np.random.choice(total_cells, cells_to_apply, replace=False)

    for cell in selected_cells:
        top_left_x = (cell % 128) * grid_width
        top_left_y = (cell // 128) * grid_height
        bottom_right_x = top_left_x + grid_width
        bottom_right_y = top_left_y + grid_height

        # Fill the selected cell with white (255)
        mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    return mask


def get_virtual_attention_map(depth_path, segmented_path, noise_cat: int = 0, central_camera: bool = True,
                              depth_threshold: float = 20.0, min_depth: float = 2.3, 
                              DILATION_KERNEL_SIZE: int = 10, DILATION_ITERATIONS: int = 1):
    segmentation = read_images(segmented_path, 'segmentation')
    if depth_path is not None:
        depth_rgb = read_images(depth_path, 'depth')
        depth_img = process_depth_image(depth_rgb)
    else:
        depth_img = None
    mask_depth, mask_segmentation, road_mask, sidewalk_mask, line_mask = create_masks(
        depth_img, segmentation, SS_CONVERTER, central_camera=central_camera,
        depth_threshold=depth_threshold, min_depth=min_depth)
    boundary = find_boundary(road_mask, sidewalk_mask, DILATION_KERNEL_SIZE, DILATION_ITERATIONS)
    boundary = cv2.bitwise_or(boundary, line_mask) * 255
    boundary = cv2.bitwise_and(mask_depth, boundary)
    merge_boundary = mask_segmentation

    merge = cv2.bitwise_and(mask_depth, merge_boundary)
    # merge_simple = merge.copy()

    if depth_img is not None:
        width, height = depth_img.shape[1], depth_img.shape[0]
        first_depth_part = depth_img < 10
        second_depth_part = (depth_img >= 10) * (depth_img < 20)
        third_depth_part = depth_img >= 20

        mask_noise_first = create_mask_noise(width, height, 70) * first_depth_part
        mask_noise_second = create_mask_noise(width, height, 50) * second_depth_part
        mask_noise_third = create_mask_noise(width, height, 10) * third_depth_part
        mask_noise = mask_noise_first + mask_noise_second + mask_noise_third

    if noise_cat == 0:
        merge = cv2.bitwise_or(merge, boundary)
    else:

        noise = generate_depth_aware_perlin_noise(depth_img, scale=3, layers=1)
        th = 0.6
        noise[noise < th] = 0
        noise[noise >= th] = 1
        noise = noise.T
        noise = 1 - noise
        # cv2.imwrite("noise.png", noise * 255)

        masked_noise_image = mask_noise * noise
        masked_noise_image = masked_noise_image.astype(np.uint8)

        # Merge with other masks
        if noise_cat == 2:
            boundary_noise = (boundary * noise).astype(np.uint8)
            # save_mask("bound.png", boundary_noise, "reprojected_images_pres_boundary_noise/")
            merge = merge * (masked_noise_image == 0) + boundary_noise
        elif noise_cat == 1:
            merge = cv2.bitwise_or(merge, boundary)
            merge = merge * (masked_noise_image == 0)
        else:
            merge = cv2.bitwise_or(merge, boundary)
            merge = (merge * noise).astype(np.uint8)

    return depth_img, segmentation, mask_depth, mask_segmentation, boundary, merge_boundary, merge


def get_virtual_noise_from_depth(depth_img, noise_cat: int = 0, txt: str = ''):
    depth_img = process_depth_image(depth_img)
    # print(depth_img)
    width, height = depth_img.shape[1], depth_img.shape[0]
    first_depth_part = depth_img < 10
    second_depth_part = (depth_img >= 10) * (depth_img < 20)
    third_depth_part = depth_img >= 20

    mask_noise_first = create_mask_noise(width, height, 70) * first_depth_part
    mask_noise_second = create_mask_noise(width, height, 50) * second_depth_part
    mask_noise_third = create_mask_noise(width, height, 10) * third_depth_part
    mask_noise = mask_noise_first + mask_noise_second + mask_noise_third

    noise = generate_depth_aware_perlin_noise(depth_img, scale=3, layers=1)
    th = 0.6
    noise[noise < th] = 0
    noise[noise >= th] = 1
    noise = noise.T
    noise = 1 - noise
    # cv2.imwrite("noise.png", noise*255)
    masked_noise_image = mask_noise * noise
    masked_noise_image = masked_noise_image.astype(np.uint8)
    # print(np.max(masked_noise_image))
    # cv2.imwrite("masked_noise_image.png", masked_noise_image)
    merge = np.ones_like(depth_img) * 255

    # Merge with other masks
    if noise_cat == 1:
        merge = merge * (masked_noise_image == 255)
    elif noise_cat == 0:
        pass
    else:
        merge = (merge * noise).astype(np.uint8)

    merge[depth_img > 30] = 255

    return merge