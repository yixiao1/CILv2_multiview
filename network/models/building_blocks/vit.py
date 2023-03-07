from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Sequence, Mapping, TypeVar, Generic
from itertools import repeat
from functools import partial
from collections import OrderedDict

from dataclasses import dataclass
from enum import Enum
from types import FunctionType

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.functional as F
import torchvision


__all__ = ['VisionTransformer', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']


# These are all Imagenet1kv1 pretrained weights
model_urls = {
    'vit_b_16': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
    'vit_b_32': 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth',
    'vit_l_16': 'https://download.pytorch.org/models/vit_l_16-852ce7e3.pth',
    'vit_l_32': 'https://download.pytorch.org/models/vit_l_32-c7638314.pth',
    'vit_h_14': 'https://download.pytorch.org/models/vit_h_14_swag-80465313.pth',
}


# class ImageClassification(nn.Module):
#     def __init__(
#         self,
#         *,
#         crop_size: int,
#         resize_size: int = 256,
#         mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
#         std: Tuple[float, ...] = (0.229, 0.224, 0.225),
#         interpolation: torchvision.transforms.InterpolationMode = torchvision.transforms.InterpolationMode.BILINEAR,
#         antialias: Optional[Union[str, bool]] = "warn",
#     ) -> None:
#         super().__init__()
#         self.crop_size = [crop_size]
#         self.resize_size = [resize_size]
#         self.mean = list(mean)
#         self.std = list(std)
#         self.interpolation = interpolation
#         self.antialias = antialias
#
#     def forward(self, img: Tensor) -> Tensor:
#         img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
#         img = F.center_crop(img, self.crop_size)
#         if not isinstance(img, Tensor):
#             img = F.pil_to_tensor(img)
#         img = F.convert_image_dtype(img, torch.float)
#         img = F.normalize(img, mean=self.mean, std=self.std)
#         return img
#
#     def __repr__(self) -> str:
#         format_string = self.__class__.__name__ + "("
#         format_string += f"\n    crop_size={self.crop_size}"
#         format_string += f"\n    resize_size={self.resize_size}"
#         format_string += f"\n    mean={self.mean}"
#         format_string += f"\n    std={self.std}"
#         format_string += f"\n    interpolation={self.interpolation}"
#         format_string += "\n)"
#         return format_string
#
#     def describe(self) -> str:
#         return (
#             "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
#             f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
#             f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
#             f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
#         )
#
#
# class ViT_B_32_Weights(WeightsEnum):
#     IMAGENET1K_V1 = Weights(
#         url="https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
#         transforms=partial(ImageClassification, crop_size=224),
#         meta={
#             **_COMMON_META,
#             "num_params": 88224232,
#             "min_size": (224, 224),
#             "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_32",
#             "_metrics": {
#                 "ImageNet-1K": {
#                     "acc@1": 75.912,
#                     "acc@5": 92.466,
#                 }
#             },
#             "_ops": 4.409,
#             "_file_size": 336.604,
#             "_docs": """
#                 These weights were trained from scratch by using a modified version of `DeIT
#                 <https://arxiv.org/abs/2012.12877>`_'s training recipe.
#             """,
#         },
#     )
#     DEFAULT = IMAGENET1K_V1
#
# # ========================== Base of Base =================================
#
#
# @dataclass
# class Weights:
#     """
#     This class is used to group important attributes associated with the pre-trained weights.
#     Args:
#         url (str): The location where we find the weights.
#         transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
#             needed to use the model. The reason we attach a constructor method rather than an already constructed
#             object is because the specific object might have memory and thus we want to delay initialization until
#             needed.
#         meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
#             informative attributes (for example the number of parameters/flops, recipe link/methods used in training
#             etc), configuration parameters (for example the `num_classes`) needed to construct the model or important
#             meta-data (for example the `classes` of a classification model) needed to use the model.
#     """
#
#     url: str
#     transforms: Callable
#     meta: Dict[str, Any]
#
#     def __eq__(self, other: Any) -> bool:
#         # We need this custom implementation for correct deep-copy and deserialization behavior.
#         # TL;DR: After the definition of an enum, creating a new instance, i.e. by deep-copying or deserializing it,
#         # involves an equality check against the defined members. Unfortunately, the `transforms` attribute is often
#         # defined with `functools.partial` and `fn = partial(...); assert deepcopy(fn) != fn`. Without custom handling
#         # for it, the check against the defined members would fail and effectively prevent the weights from being
#         # deep-copied or deserialized.
#         # See https://github.com/pytorch/vision/pull/7107 for details.
#         if not isinstance(other, Weights):
#             return NotImplemented
#
#         if self.url != other.url:
#             return False
#
#         if self.meta != other.meta:
#             return False
#
#         if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
#             return (
#                 self.transforms.func == other.transforms.func
#                 and self.transforms.args == other.transforms.args
#                 and self.transforms.keywords == other.transforms.keywords
#             )
#         else:
#             return self.transforms == other.transforms
#
#
# class WeightsEnum(Enum):
#     """
#     This class is the parent class of all model weights. Each model building method receives an optional `weights`
#     parameter with its associated pre-trained weights. It inherits from `Enum` and its values should be of type
#     `Weights`.
#     Args:
#         value (Weights): The data class entry with the weight information.
#     """
#
#     @classmethod
#     def verify(cls, obj: Any) -> Any:
#         if obj is not None:
#             if type(obj) is str:
#                 obj = cls[obj.replace(cls.__name__ + ".", "")]
#             elif not isinstance(obj, cls):
#                 raise TypeError(
#                     f"Invalid Weight class provided; expected {cls.__name__} but received {obj.__class__.__name__}."
#                 )
#         return obj
#
#     def get_state_dict(self, progress: bool) -> Mapping[str, Any]:
#         return load_state_dict_from_url(self.url, progress=progress)
#
#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}.{self._name_}"
#
#     @property
#     def url(self):
#         return self.value.url
#
#     @property
#     def transforms(self):
#         return self.value.transforms
#
#     @property
#     def meta(self):
#         return self.value.meta
#
#
# def _log_api_usage_once(obj: Any) -> None:
#
#     """
#     Logs API usage(module and name) within an organization.
#     In a large ecosystem, it's often useful to track the PyTorch and
#     TorchVision APIs usage. This API provides the similar functionality to the
#     logging module in the Python stdlib. It can be used for debugging purpose
#     to log which methods are used and by default it is inactive, unless the user
#     manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
#     Please note it is triggered only once for the same API call within a process.
#     It does not collect any data from open-source users since it is no-op by default.
#     For more information, please refer to
#     * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
#     * Logging policy: https://github.com/pytorch/vision/issues/5052;
#     Args:
#         obj (class instance or method): an object to extract info from.
#     """
#     module = obj.__module__
#     if not module.startswith("torchvision"):
#         module = f"torchvision.internal.{module}"
#     name = obj.__class__.__name__
#     if isinstance(obj, FunctionType):
#         name = obj.__name__
#     torch._C._log_api_usage_once(f"{module}.{name}")
#
#
# def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
#     """
#     Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
#     Otherwise, we will make a tuple of length n, all with value of x.
#     reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
#     Args:
#         x (Any): input value
#         n (int): length of the resulting tuple
#     """
#     if isinstance(x, collections.abc.Iterable):
#         return tuple(x)
#     return tuple(repeat(x, n))
#
#
# def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: TypeVar("V")) -> None:
#     if param in kwargs:
#         if kwargs[param] != new_value:
#             raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
#     else:
#         kwargs[param] = new_value
#
#
# class ConvNormActivation(torch.nn.Sequential):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Union[int, Tuple[int, ...]] = 3,
#         stride: Union[int, Tuple[int, ...]] = 1,
#         padding: Optional[Union[int, Tuple[int, ...], str]] = None,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         dilation: Union[int, Tuple[int, ...]] = 1,
#         inplace: Optional[bool] = True,
#         bias: Optional[bool] = None,
#         conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
#     ) -> None:
#
#         if padding is None:
#             if isinstance(kernel_size, int) and isinstance(dilation, int):
#                 padding = (kernel_size - 1) // 2 * dilation
#             else:
#                 _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
#                 kernel_size = _make_ntuple(kernel_size, _conv_dim)
#                 dilation = _make_ntuple(dilation, _conv_dim)
#                 padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
#         if bias is None:
#             bias = norm_layer is None
#
#         layers = [
#             conv_layer(
#                 in_channels,
#                 out_channels,
#                 kernel_size,
#                 stride,
#                 padding,
#                 dilation=dilation,
#                 groups=groups,
#                 bias=bias,
#             )
#         ]
#
#         if norm_layer is not None:
#             layers.append(norm_layer(out_channels))
#
#         if activation_layer is not None:
#             params = {} if inplace is None else {"inplace": inplace}
#             layers.append(activation_layer(**params))
#         super().__init__(*layers)
#         _log_api_usage_once(self)
#         self.out_channels = out_channels
#
#         if self.__class__ == ConvNormActivation:
#             warnings.warn(
#                 "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
#             )
#
#
# class Conv2dNormActivation(ConvNormActivation):
#     """
#     Configurable block used for Convolution2d-Normalization-Activation blocks.
#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
#         kernel_size: (int, optional): Size of the convolving kernel. Default: 3
#         stride (int, optional): Stride of the convolution. Default: 1
#         padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#         norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
#         activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
#         dilation (int): Spacing between kernel elements. Default: 1
#         inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
#         bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
#     """
#
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Union[int, Tuple[int, int]] = 3,
#         stride: Union[int, Tuple[int, int]] = 1,
#         padding: Optional[Union[int, Tuple[int, int], str]] = None,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         dilation: Union[int, Tuple[int, int]] = 1,
#         inplace: Optional[bool] = True,
#         bias: Optional[bool] = None,
#     ) -> None:
#
#         super().__init__(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             groups,
#             norm_layer,
#             activation_layer,
#             dilation,
#             inplace,
#             bias,
#             torch.nn.Conv2d,
#         )
#
#
# class MLP(torch.nn.Sequential):
#     """This block implements the multi-layer perceptron (MLP) module.
#     Args:
#         in_channels (int): Number of channels of the input
#         hidden_channels (List[int]): List of the hidden channel dimensions
#         norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
#         activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
#         inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
#             Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
#         bias (bool): Whether to use bias in the linear layer. Default ``True``
#         dropout (float): The probability for the dropout layer. Default: 0.0
#     """
#
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_channels: List[int],
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         inplace: Optional[bool] = None,
#         bias: bool = True,
#         dropout: float = 0.0,
#     ):
#         # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
#         # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
#         params = {} if inplace is None else {"inplace": inplace}
#
#         layers = []
#         in_dim = in_channels
#         for hidden_dim in hidden_channels[:-1]:
#             layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
#             if norm_layer is not None:
#                 layers.append(norm_layer(hidden_dim))
#             layers.append(activation_layer(**params))
#             layers.append(torch.nn.Dropout(dropout, **params))
#             in_dim = hidden_dim
#
#         layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
#         layers.append(torch.nn.Dropout(dropout, **params))
#
#         super().__init__(*layers)
#         _log_api_usage_once(self)
#
# # ========================== General Architecture =================================
#
#
# # Follow source code: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
# class ConvStemConfig(NamedTuple):
#     out_channels: int
#     kernel_size: int
#     stride: int
#     norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
#     activation_layer: Callable[..., nn.Module] = nn.ReLU
#
#
# class MLPBlock(MLP):
#     """Transformer MLP block."""
#
#     _version = 2
#
#     def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
#         super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.normal_(m.bias, std=1e-6)
#
#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         version = local_metadata.get("version", None)
#
#         if version is None or version < 2:
#             # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
#             for i in range(2):
#                 for type in ["weight", "bias"]:
#                     old_key = f"{prefix}linear_{i+1}.{type}"
#                     new_key = f"{prefix}{3*i}.{type}"
#                     if old_key in state_dict:
#                         state_dict[new_key] = state_dict.pop(old_key)
#
#         super()._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )
#
#
# class EncoderBlock(nn.Module):
#     """Transformer encoder block."""
#
#     def __init__(
#         self,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         dropout: float,
#         attention_dropout: float,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#     ):
#         super().__init__()
#         self.num_heads = num_heads
#
#         # Attention block
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#
#         # MLP block
#         self.ln_2 = norm_layer(hidden_dim)
#         self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
#
#     def forward(self, input: torch.Tensor):
#         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
#         x = self.ln_1(input)
#         x, _ = self.self_attention(x, x, x, need_weights=False)
#         x = self.dropout(x)
#         x = x + input
#
#         y = self.ln_2(x)
#         y = self.mlp(y)
#         return x + y
#
#
# class Encoder(nn.Module):
#     """Transformer Model Encoder for sequence to sequence translation."""
#
#     def __init__(
#         self,
#         seq_length: int,
#         num_layers: int,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         dropout: float,
#         attention_dropout: float,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#     ):
#         super().__init__()
#         # Note that batch_size is on the first dim because
#         # we have batch_first=True in nn.MultiAttention() by default
#         self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
#         self.dropout = nn.Dropout(dropout)
#         layers: OrderedDict[str, nn.Module] = OrderedDict()
#         for i in range(num_layers):
#             layers[f"encoder_layer_{i}"] = EncoderBlock(
#                 num_heads,
#                 hidden_dim,
#                 mlp_dim,
#                 dropout,
#                 attention_dropout,
#                 norm_layer,
#             )
#         self.layers = nn.Sequential(layers)
#         self.ln = norm_layer(hidden_dim)
#
#     def forward(self, input: torch.Tensor):
#         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
#         input = input + self.pos_embedding
#         return self.ln(self.layers(self.dropout(input)))
#
#
# class VisionTransformer(nn.Module):
#     """Vision Transformer as per https://arxiv.org/abs/2010.11929."""
#
#     def __init__(
#         self,
#         image_size: int,
#         patch_size: int,
#         num_layers: int,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         dropout: float = 0.0,
#         attention_dropout: float = 0.0,
#         num_classes: int = 1000,
#         representation_size: Optional[int] = None,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         conv_stem_configs: Optional[List[ConvStemConfig]] = None,
#     ):
#         super().__init__()
#         _log_api_usage_once(self)
#         torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.hidden_dim = hidden_dim
#         self.mlp_dim = mlp_dim
#         self.attention_dropout = attention_dropout
#         self.dropout = dropout
#         self.num_classes = num_classes
#         self.representation_size = representation_size
#         self.norm_layer = norm_layer
#
#         if conv_stem_configs is not None:
#             # As per https://arxiv.org/abs/2106.14881
#             seq_proj = nn.Sequential()
#             prev_channels = 3
#             for i, conv_stem_layer_config in enumerate(conv_stem_configs):
#                 seq_proj.add_module(
#                     f"conv_bn_relu_{i}",
#                     Conv2dNormActivation(
#                         in_channels=prev_channels,
#                         out_channels=conv_stem_layer_config.out_channels,
#                         kernel_size=conv_stem_layer_config.kernel_size,
#                         stride=conv_stem_layer_config.stride,
#                         norm_layer=conv_stem_layer_config.norm_layer,
#                         activation_layer=conv_stem_layer_config.activation_layer,
#                     ),
#                 )
#                 prev_channels = conv_stem_layer_config.out_channels
#             seq_proj.add_module(
#                 "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
#             )
#             self.conv_proj: nn.Module = seq_proj
#         else:
#             self.conv_proj = nn.Conv2d(
#                 in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
#             )
#
#         seq_length = (image_size // patch_size) ** 2
#
#         # Add a [CLS] / [CMD] token
#         self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#         seq_length += 1
#
#         self.encoder = Encoder(
#             seq_length,
#             num_layers,
#             num_heads,
#             hidden_dim,
#             mlp_dim,
#             dropout,
#             attention_dropout,
#             norm_layer,
#         )
#         self.seq_length = seq_length
#
#         heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
#         if representation_size is None:
#             heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
#         else:
#             heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
#             heads_layers["act"] = nn.Tanh()
#             heads_layers["head"] = nn.Linear(representation_size, num_classes)
#
#         self.heads = nn.Sequential(heads_layers)
#
#         if isinstance(self.conv_proj, nn.Conv2d):
#             # Init the patchify stem
#             fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
#             nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
#             if self.conv_proj.bias is not None:
#                 nn.init.zeros_(self.conv_proj.bias)
#         elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
#             # Init the last 1x1 conv of the conv stem
#             nn.init.normal_(
#                 self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
#             )
#             if self.conv_proj.conv_last.bias is not None:
#                 nn.init.zeros_(self.conv_proj.conv_last.bias)
#
#         if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
#             fan_in = self.heads.pre_logits.in_features
#             nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
#             nn.init.zeros_(self.heads.pre_logits.bias)
#
#         if isinstance(self.heads.head, nn.Linear):
#             nn.init.zeros_(self.heads.head.weight)
#             nn.init.zeros_(self.heads.head.bias)
#
#     def _process_input(self, x: torch.Tensor) -> torch.Tensor:
#         n, c, h, w = x.shape
#         p = self.patch_size
#         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
#         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
#         n_h = h // p
#         n_w = w // p
#
#         # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
#         x = self.conv_proj(x)
#         # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
#         x = x.reshape(n, self.hidden_dim, n_h * n_w)
#
#         # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
#         # The self attention layer expects inputs in the format (N, S, E)
#         # where S is the source sequence length, N is the batch size, E is the
#         # embedding dimension
#         x = x.permute(0, 2, 1)
#
#         return x
#
#     def forward(self, x: torch.Tensor):
#         # Reshape and permute the input tensor
#         x = self._process_input(x)
#         n = x.shape[0]
#
#         # Expand the class token to the full batch
#         batch_class_token = self.class_token.expand(n, -1, -1)
#         x = torch.cat([batch_class_token, x], dim=1)
#
#         x = self.encoder(x)
#
#         # Classifier "token" as used by standard language architectures
#         x = x[:, 0]
#
#         x = self.heads(x)
#
#         return x
#
# def _vision_transformer(
#         patch_size: int,
#         num_layers: int,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         weights: Optional[WeightsEnum],
#         progress: bool,
#         **kwargs: Any,
# ) -> VisionTransformer:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
#         assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
#         _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
#     image_size = kwargs.pop("image_size", 224)
#
#     model = VisionTransformer(
#         image_size=image_size,
#         patch_size=patch_size,
#         num_layers=num_layers,
#         num_heads=num_heads,
#         hidden_dim=hidden_dim,
#         mlp_dim=mlp_dim,
#         **kwargs,
#     )
#
#     if weights:
#         model.load_state_dict(weights.get_state_dict(progress=progress))
#
#     return model


# ========================== Models =================================

def vit_b_16(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = torchvision.models.vit_b_16(pretrained=pretrained, **kwargs)
    if pretrained:
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)
    return model


def vit_b_32(pretrained=False, **kwargs: Any):
    """Constructs a ViT-B/32 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet-1k
    """
    model = _vision_transformer(patch_size=32, num_layers=12, num_heads=12, hidden_dim=768,
                                mlp_dim=3072, weights=None, progress=False, **kwargs)
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['vit_b_32'])
        # remove the fc layers
        del model_dict['heads.head.weight']
        del model_dict['heads.head.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)

    return model


def vit_l_16(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet50'])
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)

    return model


def vit_l_32(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        model_dict = model_zoo.load_url(model_urls['resnet101'])
        # remove the fc layers to adapt to our own classes, avoiding size mismatch
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)

    return model


def vit_h_14(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
