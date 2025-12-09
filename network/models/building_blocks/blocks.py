from typing import List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi

    # Frequency buffer
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    
    # Split channels
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


def gen_sineembed_for_action(act_tensor: torch.Tensor, hidden_dim: int = 66):
    """
    Sine/cosine embedding for ACTION sequences (throttle, steer, brake), analogous to
    gen_sineembed_for_position(pos_tensor,hidden_dim).

    Args:
        act_tensor : tensor of shape [..., 3] with channels:
                     act[..., 0] = throttle (normalized to [-1,1] or [0,1])
                     act[..., 1] = steer    ([-1,1])
                     act[..., 2] = brake    ([0,1])
        hidden_dim : total embedding dimension. MUST be divisible by 6 so that each of the
                     3 channels gets an even number of frequency terms (for sin/cos pairing).

    Returns:
        Tensor of shape [..., hidden_dim] with concatenated embeddings for
        (throttle || steer || brake), in that order.

    Notes:
        - Uses the same scaling and frequency schedule as your position PE:
              scale = 2 * pi
              dim_t = 10000 ** (2 * (idx // 2) / per_channel_dim)
        - Mirrors the pattern:
              stack( sin(even_idx), cos(odd_idx) ).flatten(-2)
          so per-channel dimension must be even.
    """
    # if act_tensor.shape[-1] != 3:
    #     raise ValueError(f"gen_sineembed_for_action expects last dim=3, got {act_tensor.shape[-1]}")
    # if hidden_dim % 6 != 0:
    #     raise ValueError(f"hidden_dim must be divisible by 6; got {hidden_dim}")

    # per_ch_dim = hidden_dim // 3  # chunk per channel
    # if per_ch_dim % 2 != 0:
    #     # This should never happen if hidden_dim % 6 == 0, but keep a guard.
    #     raise ValueError(f"hidden_dim/3 must be even; got per_ch_dim={per_ch_dim}")

    per_ch_dim = hidden_dim // 3
    scale = 2 * math.pi
    dtype = act_tensor.dtype

    # Frequency buffer like in gen_sineembed_for_position
    dim_t = torch.arange(per_ch_dim, dtype=torch.float32, device=act_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / per_ch_dim)  # [per_ch_dim]

    # Split channels
    th = (act_tensor[..., 0] * scale).to(dtype)  # throttle
    st = (act_tensor[..., 1] * scale).to(dtype)  # steer
    br = (act_tensor[..., 2] * scale).to(dtype)  # brake

    # Throttle embeddings
    # ch: [...], expand with frequency axis
    th_ch_div = th[..., None] / dim_t  # [..., per_ch_dim]
    th_sin_part = th_ch_div[..., 0::2].sin()
    th_cos_part = th_ch_div[..., 1::2].cos()
    # If per_ch_dim is even, shapes match; stack then flatten to [..., per_ch_dim]
    th_emb = torch.stack((th_sin_part, th_cos_part), dim=-1).flatten(-2) # [..., per_ch_dim]

    # Steering embeddings
    # ch: [...], expand with frequency axis
    st_ch_div = st[..., None] / dim_t  # [..., per_ch_dim]
    st_sin_part = st_ch_div[..., 0::2].sin()
    st_cos_part = st_ch_div[..., 1::2].cos()
    # If per_ch_dim is even, shapes match; stack then flatten to [..., per_ch_dim]
    st_emb = torch.stack((st_sin_part, st_cos_part), dim=-1).flatten(-2) # [..., per_ch_dim]

    # Brake embeddings
    # ch: [...], expand with frequency axis
    br_ch_div = br[..., None] / dim_t  # [..., per_ch_dim]
    br_sin_part = br_ch_div[..., 0::2].sin()
    br_cos_part = br_ch_div[..., 1::2].cos()
    # If per_ch_dim is even, shapes match; stack then flatten to [..., per_ch_dim]
    br_emb = torch.stack((br_sin_part, br_cos_part), dim=-1).flatten(-2) # [..., per_ch_dim]

    # Concatenate in a fixed order: throttle || steer || brake
    emb = torch.cat((th_emb, st_emb, br_emb), dim=-1)  # [..., hidden_dim]
    return emb


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GridSampleCrossBEVAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=8, config=None):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims  # E (transformer width, e.g. 256)
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points  # T (trajectory length, e.g. 8)
        self.config = config
        self.attention_weights = nn.Linear(embed_dims,num_points)   # [B, 20, 256] → [B, 20, 8]
        self.output_proj = nn.Linear(embed_dims, embed_dims)  # [B, 20, 256] → [B, 20, 256]
        self.dropout = nn.Dropout(0.1)


        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True), # [B, 256, H, W] → [B, 256, H, W]
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Args:
            queries:       [B, 20, 256]
            traj_points:   [B, 20, 8, 2]   (x,y) in meters (ego frame)
            bev_feature:   [B, 256, H, W]
            spatial_shape: (H, W) (unused here)
        """

        bs, num_queries, num_points, _ = traj_points.shape  # B, 20, 8, 2
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = traj_points.clone()  # [B, 20, 8, 2]
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y # x / Ymax  → [B, 20, 8]
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x # y / Xmax  → [B, 20, 8]

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y: [B, 20, 8, 2]
        
        # ---- 2) Predict weights over T points for each query ----
        attention_weights = self.attention_weights(queries)  # [B, 20, 256] → [B, 20, 8]
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)  # [B, 20, 8]

        # ---- 3) Project feature map ----
        value = self.value_proj(bev_feature)  # [B, 256, H, W] → [B, 256, H, W]

        # ---- 4) Grid sample features along trajectory ----
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)  # [B, 20, 8, 2]
        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value,   # [B, 256, H, W]
            grid,   # [B, 20, 8, 2]
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # [B, 256, 20, 8]

        # ---- 5) Weighted sum over trajectory points ----
        attention_weights = attention_weights.unsqueeze(1)  # [B, 1, 20, 8]
        out = (attention_weights * sampled_features).sum(dim=-1)  # [B, 256, 20]
        out = out.permute(0, 2, 1).contiguous()  # [B, 20, 256]
        
        # ---- 6) Project + residual ---- | this projection is analogous to the W^O projection in the original transformer.
        out = self.output_proj(out)  # [B, 20, 256]

        return self.dropout(out) + queries  # [B, 20, 256]


class DeformableTrajectoryCrossAttention(nn.Module):
    """
    Inspired in MSDeformAttn cross-attention (https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py) that aggregates features sampled
    along given trajectory points on a 2D feature map.

    References:
      - "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
        (ICLR 2021). We follow its spirit: per-head weights over a small set of
        bilinearly sampled points from a CNN feature map. Here, the sampling
        points are provided by the planning trajectory instead of learned offsets.

    Args:
        embed_dims (int): token/feature channel dimension E.
        num_heads (int): number of attention heads H (E must be divisible by H).
        num_levels (int): kept for API compatibility (unused: single-scale here).
        in_cnn_dims (int): input feature map channels C_in (we project to E).
        num_points (int): number of temporal points per trajectory (T).
        config (Any): config object. If it has lidar_max_x / lidar_max_y, we use
                      them to normalize coords when normalization='range'.
        normalization (str): {'range','hw','none'}
            - 'range' (default): expects traj_points in metric coords; normalizes
              with (x / config.lidar_max_x, y / config.lidar_max_y), then swaps
              to (u,v) for grid_sample and rescales to [-1,1].
            - 'hw': expects traj_points in pixel coords (x in [0,W), y in [0,H)),
              normalizes to [-1,1] using H,W.
            - 'none': traj_points are already normalized in [-1,1] grid coords (u,v).
        dropout (float): dropout prob.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 num_levels: int = 1,
                 in_cnn_dims: int = 64,
                 num_points: int = 8,
                 config: Optional[object] = None,
                 normalization: str = 'range',
                 dropout: float = 0.1):
        super().__init__()
        assert embed_dims % num_heads == 0, "embed_dims must be divisible by num_heads"
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.normalization = normalization

        head_dim = embed_dims // num_heads

        # Project the 2D feature map to embed_dims (E), like MSDeformAttn's value proj.
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_cnn_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Per-head attention weights over T sampling points, predicted from query
        # (analogous to MSDeformAttn's A_{qhk}).
        self.attn_weights = nn.Linear(embed_dims, num_heads * num_points)

        # Output projection (like MSDeformAttn's final linear)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Following common transformer init conventions
        nn.init.xavier_uniform_(self.value_proj[0].weight)
        nn.init.constant_(self.value_proj[0].bias, 0.)

        nn.init.constant_(self.attn_weights.weight, 0.)
        nn.init.constant_(self.attn_weights.bias, 0.)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    @staticmethod
    def _normalize_hw(traj_points: torch.Tensor,
                      spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Normalize pixel coords to [-1,1] grid for grid_sample.
        traj_points: [B, M, T, 2] with (x,y) in pixel coords: x in [0,W), y in [0,H)
        returns grid in (u,v) with u for W (cols), v for H (rows).
        """
        H, W = spatial_shape
        # Convert (x,y) pixels -> [-1,1]
        u = (traj_points[..., 0] / max(W - 1, 1) - 0.5) * 2.0
        v = (traj_points[..., 1] / max(H - 1, 1) - 0.5) * 2.0
        grid = torch.stack([u, v], dim=-1)
        return grid

    def _normalize_range(self, traj_points: torch.Tensor) -> torch.Tensor:
        """
        Normalize metric ego coords using config.lidar_max_x / lidar_max_y and
        convert to grid_sample order (u,v). Expects traj_points[...,0]=x (forward),
        traj_points[...,1]=y (lateral). We map to (u,v) = (y/x-range, x/y-range).
        """
        if self.config is None or not hasattr(self.config, 'lidar_max_x') or not hasattr(self.config, 'lidar_max_y'):
            raise ValueError(
                "normalization='range' requires config.lidar_max_x and config.lidar_max_y."
            )
        # Normalize to [-1,1]; swap to (u,v) to match grid_sample's (x->W, y->H)
        v = traj_points[..., 0] / float(self.config.lidar_max_y)  # NOTE: keep naming aligned with your former code
        u = traj_points[..., 1] / float(self.config.lidar_max_x)
        grid = torch.stack([u, v], dim=-1).clamp_(-1.0, 1.0)
        return grid

    def forward(self,
                queries: torch.Tensor,            # [B, M, E]
                traj_points: torch.Tensor,        # [B, M, T, 2] (coords; normalization depends on self.normalization)
                bev_feature: torch.Tensor,        # [B, C_in, H, W] (any 2D spatial feature map; not necessarily BEV)
                spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Returns:
            out: [B, M, E]  (residual-updated queries)
        """
        B, M, E = queries.shape
        assert E == self.embed_dims
        assert traj_points.shape[:3] == (B, M, self.num_points)

        # 1) Project values and reshape into heads
        #    value: [B, E, H, W] -> [B, Hh, Dh, H, W]
        value = self.value_proj(bev_feature)
        Hh = self.num_heads
        Dh = self.embed_dims // self.num_heads
        value = value.view(B, Hh, Dh, value.shape[2], value.shape[3])  # [B, Hh, Dh, H, W]

        # 2) Build sampling grid in [-1,1] for grid_sample (u for width, v for height)
        if self.normalization == 'none':
            # assume traj_points already in [-1,1] with order (u,v)
            grid = traj_points
        elif self.normalization == 'hw':
            if spatial_shape is None:
                spatial_shape = (bev_feature.shape[2], bev_feature.shape[3])
            grid = self._normalize_hw(traj_points, spatial_shape)
        else:
            # 'range' (default): normalize with config.{lidar_max_x, lidar_max_y}
            grid = self._normalize_range(traj_points)

        # grid_sample expects [B, H_out, W_out, 2]; we hack H_out=M, W_out=T
        # to get [B, C, M, T] output.
        grid = grid.view(B, M, self.num_points, 2)

        # 3) Sample features per head
        #    We need to sample per head; grid_sample runs on [B, C, H, W].
        #    So we fold heads into C and unfold back.
        value_merged = value.view(B, Hh * Dh, value.shape[-2], value.shape[-1])  # [B, E, H, W]
        sampled = F.grid_sample(
            value_merged, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )  # [B, E, M, T]
        sampled = sampled.view(B, Hh, Dh, M, self.num_points)  # [B, Hh, Dh, M, T]

        # 4) Per-head attention weights over T points from queries
        attn = self.attn_weights(queries)  # [B, M, Hh*T]
        attn = attn.view(B, M, Hh, self.num_points).softmax(dim=-1)  # [B, M, Hh, T]

        # 5) Weighted sum over T, then merge heads -> [B, M, E]
        #    Align dims to multiply: sampled [B,Hh,Dh,M,T], attn [B,M,Hh,T]
        sampled = sampled.permute(0, 3, 1, 2, 4)          # [B, M, Hh, Dh, T]
        attn = attn.permute(0, 1, 2, 3).unsqueeze(3)      # [B, M, Hh, 1, T]
        fused = (attn * sampled).sum(dim=-1)              # [B, M, Hh, Dh]
        fused = fused.reshape(B, M, Hh * Dh)              # [B, M, E]

        # 6) Output projection + residual
        out = self.output_proj(fused)
        out = self.dropout(out) + queries
        return out


