from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval

import torch.distributed

from configs.attribute_dict import AttributeDict
import copy
import numpy as np
import os
import yaml

from logger._logger import create_log

_g_conf = AttributeDict()

_g_conf.immutable(False)

"""#### GENERAL CONFIGURATION PARAMETERS ####"""
"""#### Training Related Parameters"""
_g_conf.MAGICAL_SEED = 0
_g_conf.NUM_WORKER = 14
_g_conf.DATA_PARALLEL = False
_g_conf.USE_AUTOCAST = False
_g_conf.TRAINING_RESUME = True
_g_conf.FINETUNE = False
_g_conf.FINETUNE_MODEL = ''
_g_conf.BATCH_SIZE = 120
_g_conf.NUMBER_EPOCH = 100     # Total number of training epochs
_g_conf.SUBSET_SIZE = 1.0      # Percentage of the dataset to use for training (between 0 and 1)
_g_conf.TRAIN_DATASET_NAME = []
_g_conf.VALID_DATASET_NAME = []      # More than one datasets could be evaluated, thus a list
_g_conf.GT_DATA_USED = 'il_data'
_g_conf.DATA_USED = ['rgb_left', 'rgb_central', 'rgb_right']
_g_conf.DATA_INFORMATION = {'rgb_left': {'fov': 60, 'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]},
                            'rgb_central': {'fov': 60, 'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]},
                            'rgb_right': {'fov': 60, 'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]}}
_g_conf.LENS_CIRCLE_SET = False
_g_conf.IMAGE_SHAPE = [3, 88, 200]
_g_conf.ENCODER_INPUT_FRAMES_NUM = 1
_g_conf.ENCODER_STEP_INTERVAL = 1     # the pace step of frame you want to use. For example, if you want to have 5 sequential input images taking pre 20-frames as a step, you should set INPUT_FRAMES_NUM =5 and INPUT_FRAME_INTERVAL=20
_g_conf.ENCODER_OUTPUT_STEP_DELAY = 0  # whether we want to predict the future data points or just the current point
_g_conf.DECODER_OUTPUT_FRAMES_NUM = 1
_g_conf.AUGMENTATION = False
_g_conf.COLOR_JITTER = False
_g_conf.AUG_MIX = False
_g_conf.RAND_AUGMENT = False
_g_conf.DATA_FPS = 10
_g_conf.DATA_COMMAND_CLASS_NUM = 4
_g_conf.DATA_COMMAND_ONE_HOT = True
_g_conf.SPEED_AUGMENTATION = False
_g_conf.SPEED_AUGMENTATION_MIN_ACC = 0.04  # m/s2
_g_conf.SPEED_AUGMENTATION_PROB = 0.3  # max = 1.0
_g_conf.SPEED_AUGMENTATION_MIN_PERC = 0.3  # Min decrease percentage of the original speed value; max = 1.0
_g_conf.ERROR_CAM_AUGMENTATION = False
_g_conf.ERROR_LIST_CAM_AUGMENTATION = ['conti_front']
_g_conf.ERROR_CAM_AUGMENTATION_PROB = 0.1  # max = 1.0
_g_conf.DATA_NORMALIZATION = {}
_g_conf.IMG_NORMALIZATION = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}   # ImageNet by default
_g_conf.VIRTUAL_ATTENTION_INTERPOLATION = 'INTER_LINEAR'  # INTER_LINEAR, INTER_AREA, or INTER_NEAREST (others aren't worth it)
_g_conf.BINARIZE_ATTENTION = False  # Binarize the attention map (attmap[attmap > 0] = 1)
_g_conf.FC_LAYER_NORM = False  # Use LayerNorm on the FC layers
_g_conf.EXP_SAVE_PATH = '_results'
_g_conf.TARGETS = ['steer', 'throttle', 'brake']  # From the float data, the ones that the network should estimate
_g_conf.ACCELERATION_AS_ACTION = False
_g_conf.OTHER_INPUTS = ['speed']  # From the float data, the ones that are input to the neural network
_g_conf.ACTION_TOKEN = False
_g_conf.SPEED_TOKEN = False
_g_conf.MASK_DIAGONAL_ATTMAP = False
_g_conf.ACCEL_ROT_AS_INPUT = False  # Use the accelerometer data as input to the model
_g_conf.ADAPTIVE_QUANTILE_REGRESSION = False  # Use AQR for acceleration loss
_g_conf.QUANTILE_REGRESSION_SCHEDULE = False  # Schedule AQR for acceleration loss
_g_conf.ADAPTIVE_QUANTILE_REGRESSION_TARGET = 0.5  # Target quantile for AQR

"""#### Optimizer Related Parameters ####"""
_g_conf.LOSS = ''    # It can be the name of loss, such as L1, CrossEntropy, or an architecure name such as "fasterRcnn, deeplabv3", which means we use the same loss as this architectures
_g_conf.LOSS_POW = 1
_g_conf.LOSS_WEIGHT = {}
_g_conf.ATTENTION_LOSS = False  # Use loss on the attention maps (must have a ground truth, virtual or real)
_g_conf.MHA_ATTENTION_COSSIM_LOSS = False  # Apply a loss to each head of the MHA block in the Transformer Encoder
_g_conf.LEARNING_RATE = 0.0002       # the max learning rate setting
_g_conf.LEARNING_RATE_SCHEDULE = 'step'  # the learning rate schedule; 'step' -> StepLR, 'warmup_cooldown' -> linear warmup, cosine cooldown
_g_conf.LEARNING_RATE_DECAY = True
_g_conf.LEARNING_RATE_MINIMUM = 0.00001
_g_conf.LEARNING_RATE_DECAY_EPOCHES = []    # we adjust learning rate for each 1000 iterations
_g_conf.LEARNING_RATE_POLICY = {'name': 'normal', 'level': 0.5, 'momentum': 0, 'weight_decay': 0}   # lr multiply by 0.5 for each LEARNING_RATE_STEP
_g_conf.AUTOCAST = False
_g_conf.SAVE_FULL_STATE = True

"""#### Network Related Parameters ####"""
_g_conf.MODEL_TYPE = ''
_g_conf.MODEL_CONFIGURATION = {}
_g_conf.IMAGENET_PRE_TRAINED = False
_g_conf.LOAD_CHECKPOINT = ''
_g_conf.LEARNABLE_POS_EMBED = False
_g_conf.ONE_ACTION_TOKEN = False
_g_conf.PRETRAINED_ACC_STR_TOKENS = False # If two action tokens are used, we can use start them from the pre-trained CLS token
_g_conf.NO_ACT_TOKENS = False  # No tokens will be used for the action prediction
_g_conf.PRETRAINED_ACT_TOKENS = False  # If only one action token is used, we can start it from the pre-trained CLS token
_g_conf.CMD_SPD_TOKENS = False  # Instead of adding the speed and action embeddings to the sequence, we can concatenate them to the sequence as tokens
_g_conf.PREDICT_CMD_SPD = False  # Predict the input command and speed
_g_conf.FREEZE_CLS_TOKEN = False  # freeze the classification token
_g_conf.REMOVE_CLS_TOKEN = True  # remove the classification token from the sequence
_g_conf.SENSOR_EMBED = False  # whether to use sensor embedding as in InterFuser/ReasonNet
_g_conf.EXTRA_POS_EMBED = False  # A final positional embedding at the output of the encoder
_g_conf.OLD_TOKEN_ORDER = True  # Originally had mixed the order of the [STR] and [ACC]; should be set explicitly
_g_conf.NEW_COMMAND_SPEED_FC = False  # Originally had a FC layer for the command and speed embeddings; should be set explicitly
_g_conf.LEARNABLE_ACTION_RATIO = False  # Ratio of actions between patches and specialized action tokens is learnable
_g_conf.EARLY_COMMAND_SPEED_FUSION = True  # Fusion of the command and speed embeddings is done before the camera encoder
_g_conf.LATE_COMMAND_SPEED_FUSION = False  # Fusion of the command and speed embeddings is done before each steering and acceleration encoders
_g_conf.NUM_REGISTER_TOKENS = 0  # From: https://arxiv.org/abs/2309.16588
_g_conf.EARLY_ATTENTION = False  # False for late attention (Tf. Enc. attention maps), True for early attention (resnet attention maps)
_g_conf.TFX_ENC_ATTENTION_LAYER = -1  # The layer of the encoder to use for the attention maps
_g_conf.RN_ATTENTION_LAYER = -1  # The block of the ResNet to use for the attention maps
_g_conf.ATTENTION_AS_INPUT = False  # Use the attention maps as part of the input to the model
_g_conf.ATTENTION_AS_NEW_CHANNEL = True  # If the attention maps are used as input, they will be concatenated as a new channel; else, multiply element-wise the RGB values
_g_conf.ATTENTION_FROM_UNET = False  # Attention maps will come from a UNet prediction
_g_conf.ATTENTION_NOISE_CATEGORY = 0  # 0 = no noise, if > 0, the attention map is noisy (randomly, for driving with output from U-Net)

"""#### Validation Related Parameters"""
_g_conf.EVAL_SAVE_LAST_ATT_MAPS = True  # Save the attention map of the last layer of the Encoder
_g_conf.EVAL_SAVE_LAST_Conv_ACTIVATIONS = False     # the last Conv. layer of backbone that to be saved attention maps
_g_conf.EVAL_BATCH_SIZE = 1          # batch size for evaluation
_g_conf.EVAL_SAVE_EPOCHES = [1]      # we specify the epoch we want to do offline evaluation
_g_conf.EVAL_IMAGE_WRITING_NUMBER = 10
_g_conf.EARLY_STOPPING = False          # By default, we do not apply early stopping
_g_conf.EARLY_STOPPING_PATIENCE = 3
_g_conf.EVAL_DRAW_OFFLINE_RESULTS_GRAPHS = None

"""#### Logger Related Parameters"""
_g_conf.TRAIN_LOG_SCALAR_WRITING_FREQUENCY = 2
_g_conf.TRAIN_IMAGE_WRITING_NUMBER = 2
_g_conf.TRAIN_IMAGE_LOG_FREQUENCY = 1000
_g_conf.TRAIN_PRINT_LOG_FREQUENCY = 25


def merge_with_yaml(yaml_filename, process_type='train_val'):
    """Load a yaml config file and merge it into the global config object"""
    global _g_conf
    with open(yaml_filename, 'r') as f:

        yaml_file = yaml.safe_load(f)

        yaml_cfg = AttributeDict(yaml_file)

    path_parts = os.path.split(yaml_filename)
    if process_type == 'train_val':
        _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
        _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]
    if process_type == 'val_only':
        _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
        _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]
    elif process_type == 'drive':
        _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(os.path.split(path_parts[-2])[-2])[-1]
        _g_conf.EXPERIMENT_NAME = os.path.split(path_parts[-2])[-1]
    _merge_a_into_b(yaml_cfg, _g_conf)


def get_names(folder):
    alias_in_folder = os.listdir(os.path.join('configs', folder))

    experiments_in_folder = {}
    for experiment_alias in alias_in_folder:

        g_conf.immutable(False)
        merge_with_yaml(os.path.join('configs', folder, experiment_alias))

        experiments_in_folder.update({experiment_alias: g_conf.EXPERIMENT_GENERATED_NAME})

    return experiments_in_folder


def create_exp_path(root, experiment_batch, experiment_name):
    # This is hardcoded the logs always stay on the _logs folder
    root_path = os.path.join(root, '_results')
    if not os.path.exists(os.path.join(root_path, experiment_batch, experiment_name)):
        os.makedirs(os.path.join(root_path, experiment_batch, experiment_name))


def set_type_of_process(process_type, root, rank=0, ddp=False):
    """
    This function is used to set which is the type of the current process, test, train or val
    and also the details of each since there could be many vals and tests for a single
    experiment.

    NOTE: AFTER CALLING THIS FUNCTION, THE CONFIGURATION CLOSES

    Args:
        type:

    Returns:

    """

    if process_type == 'train_val' or process_type == 'train_only' or process_type == 'val_only' \
            or process_type == 'drive':
        _g_conf.PROCESS_NAME = process_type
        if not os.path.exists(os.path.join(root, '_results', _g_conf.EXPERIMENT_BATCH_NAME, _g_conf.EXPERIMENT_NAME,
                                           'checkpoints')) and rank == 0:
            os.mkdir(os.path.join(root, '_results', _g_conf.EXPERIMENT_BATCH_NAME,
                                  _g_conf.EXPERIMENT_NAME,
                                  'checkpoints'))
        if process_type != 'drive' and ddp:
            torch.distributed.barrier()
        _g_conf.EXP_SAVE_PATH = os.path.join(root, '_results', _g_conf.EXPERIMENT_BATCH_NAME, _g_conf.EXPERIMENT_NAME)

    else:
        raise ValueError("Not found type of process")

    if (process_type == 'train_val' or process_type == 'train_only' or process_type == 'val_only') and rank == 0:
        create_log(_g_conf.EXP_SAVE_PATH, _g_conf.TRAIN_LOG_SCALAR_WRITING_FREQUENCY, _g_conf.TRAIN_IMAGE_LOG_FREQUENCY)

    _g_conf.immutable(True)


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary 'a' into config dictionary 'b', clobbering the
    options in 'b' whenever they are also specified in 'a'.
    """

    assert isinstance(a, AttributeDict) or isinstance(a, dict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttributeDict) or isinstance(a, dict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if is it more than second stack
            if stack is not None:
                b[k] = v_
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)

        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts

        b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects

    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #

    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, type(None)):
        value_a = value_a
    elif isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_b, range) and not isinstance(value_a, list):
        value_a = eval(value_a)
    elif isinstance(value_b, range) and isinstance(value_a, list):
        value_a = list(value_a)
    elif isinstance(value_b, dict):
        value_a = eval(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


g_conf = _g_conf
