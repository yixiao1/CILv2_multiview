import importlib

import torch
import torch.nn as nn
from einops import rearrange, reduce

from configs import g_conf
from _utils import utils

from network.models.building_blocks.PositionalEncoding import PositionalEncoding
from network.models.building_blocks.vit import Encoder


class CIL_vit(nn.Module):
    def __init__(self, params, rank: int = 0):
        super(CIL_vit, self).__init__()
        self.params = params

        # Get ViT model characteristics (our new perception module)
        vit_module = importlib.import_module('network.models.building_blocks.vit')
        vit_module = getattr(vit_module, self.params['encoder_embedding']['perception']['vit']['name'])
        vit = vit_module(pretrained=g_conf.IMAGENET_PRE_TRAINED)

        self.image_channels, self.image_height, self.image_width = g_conf.IMAGE_SHAPE
        self.num_cameras = len(g_conf.DATA_USED)

        # Get the vision transformer characteristics
        self.camera_tfx_hidden_dim = self.steering_tfx_hidden_dim = self.accel_tfx_hidden_dim = vit.hidden_dim  # D
        self.camera_tfx_patch_size = vit.patch_size  # P
        self.camera_tfx_num_patches_h = self.image_height // self.camera_tfx_patch_size  # H/P
        self.camera_tfx_num_patches_w = self.image_width // self.camera_tfx_patch_size  # W/P
        self.camera_tfx_num_patches = self.camera_tfx_num_patches_h * self.camera_tfx_num_patches_w  # (H/P)*(W/P)

        # Camera Transformer Encoder
        self.camera_tfx_conv_projection = vit.conv_proj
        self.camera_tfx_encoder = vit.encoder

        self.camera_tfx_num_layers = vit.encoder.num_layers
        self.camera_tfx_dropout = vit.encoder.dropout
        self.camera_tfx_num_heads = vit.encoder.num_heads

        # Sequence length for the three Encoders
        self.camera_tfx_seq_length = self.camera_tfx_num_patches + 2  # K=2 for the additional tokens
        self.steering_tfx_seq_length = len(g_conf.DATA_USED) + 1  # K=1 for the final steering token
        self.accel_tfx_seq_length = len(g_conf.DATA_USED) + 1  # K=1 for the final acceleration token

        # Steering Transformer Encoder
        self.steering_tfx_encoder = Encoder(seq_length=self.steering_tfx_seq_length,
                                            num_layers=4,
                                            num_heads=4,
                                            hidden_dim=self.camera_tfx_hidden_dim,
                                            mlp_dim=4 * self.camera_tfx_hidden_dim,
                                            dropout=0.0, attention_dropout=0.0)
        self.pos_embed_steering_tfx = nn.Parameter(torch.zeros(1, self.steering_tfx_seq_length, self.steering_tfx_hidden_dim))

        # Acceleration Transformer Encoder
        self.accel_tfx_encoder = Encoder(seq_length=self.accel_tfx_seq_length,
                                         num_layers=4,
                                         num_heads=4,
                                         hidden_dim=self.camera_tfx_hidden_dim,
                                         mlp_dim=4 * self.camera_tfx_hidden_dim,
                                         dropout=0.0, attention_dropout=0.0)
        self.pos_embed_accel_tfx = nn.Parameter(torch.zeros(1, self.accel_tfx_seq_length, self.accel_tfx_hidden_dim))

        # Final tokens
        self.final_steer_token = nn.Parameter(torch.empty(1, 1, self.steering_tfx_hidden_dim).normal_(std=0.02))
        self.final_accel_token = nn.Parameter(torch.empty(1, 1, self.accel_tfx_hidden_dim).normal_(std=0.02))

        # Command and speed embedding
        self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, self.camera_tfx_hidden_dim)
        self.speed = nn.Linear(1, self.camera_tfx_hidden_dim)

        # Steering and Acceleration tokens (STR and ACC) for the camera transformer
        if g_conf.PRETRAINED_ACC_STR_TOKENS:
            # Start from the [CLS] token of the pretrained model
            print('Initializing the acceleration ([ACC]) and steering ([STR]) tokens with the [CLS] token...') if rank == 0 else None
            self.camera_tfx_steer_token = nn.Parameter(vit.class_token.detach().clone())
            self.camera_tfx_accel_token = nn.Parameter(vit.class_token.detach().clone())
        else:
            # Start from scratch
            print('Randomly initializing the acceleration ([ACC]) and steering ([STR]) tokens...')
            self.camera_tfx_steer_token = nn.Parameter(torch.empty(1, 1, self.camera_tfx_hidden_dim).normal_(std=0.02))
            self.camera_tfx_accel_token = nn.Parameter(torch.empty(1, 1, self.camera_tfx_hidden_dim).normal_(std=0.02))

        self.act_tokens_pos = [0, 1]  # Which tokens in the sequence to use for the action

        # TODO: Add the command and speed as tokens to the sequence
        if g_conf.CMD_SPD_TOKENS:
            self.camera_tfx_seq_length += 2
            self.act_tokens_pos = [_ + 2 for _ in self.act_tokens_pos]

        # Positional embedding/encoding for the camera transformer
        if g_conf.LEARNABLE_POS_EMBED:
            self.pos_embed_camera_tfx = nn.Parameter(torch.zeros(1, self.camera_tfx_seq_length,
                                                                 self.camera_tfx_hidden_dim))
        else:
            self.pos_embed_camera_tfx = PositionalEncoding(d_model=self.camera_tfx_hidden_dim,
                                                           max_len=self.camera_tfx_seq_length)

        # We don't want to undo the pretrained weights of the ViT!
        for name, module in self.named_modules():
            if not name.startswith(('layers.encoder', 'encoder.layers')):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.1)

        self.train()

    def encode_observations(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]  # Batch size
        cam = len(g_conf.DATA_USED)  # Number of cameras/views

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = rearrange(x, 'B S cam C H W -> (B S cam) C H W')  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        # First, patch and embed the input images
        e_p = self.tfx_conv_projection(x)  # [B*S*cam, D, H/P, W/P]; usually H = W = 224
        e_p = rearrange(e_p, '(B S cam) D (patches_h) (patches_w) -> (B S cam) (patches_h patches_w) D',
                        B=B, cam=cam)  # [B*S*cam, H*W/P^2, D]

        # Set up the first tokens in the sequence
        n = e_p.shape[0]  # B*S*cam
        first_tokens = [self.camera_tfx_steer_token.expand(n, -1, -1),
                        self.camera_tfx_accel_token.expand(n, -1, -1)]

        # Concatenate the first tokens to the image embeddings
        e_p = torch.cat([*first_tokens, e_p], dim=1)  # [B*S*cam, (H/P)(W/P) + K, D]]), K = 3 if w/CLS token else 2

        # Embedding of command and speed
        e_d = self.command(d).unsqueeze(1)  # [B, 1, D]
        e_s = self.speed(s).unsqueeze(1)  # [B, 1, D]

        e_d = e_d.repeat(S * cam, 1, 1)  # [B*S*cam, 1, D]
        e_s = e_s.repeat(S * cam, 1, 1)  # [B*S*cam, 1, D]

        # Add the embeddings to the image embeddings
        if g_conf.CMD_SPD_TOKENS:
            encoded_obs = torch.cat([e_p, e_d, e_s], dim=1)  # [B*S*cam, (H//P)^2 + K, D]; K = 5 w/CLS token else 4
        else:
            encoded_obs = e_p + e_d + e_s  # [B*S*cam, (H//P)^2 + K, D]; K = 3 w/CLS token else 2

        return encoded_obs

    def action_prediction(self, sequence):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        cam = len(g_conf.DATA_USED)  # Number of cameras/views
        B = sequence.shape[0] // (S * cam)  # Batch size

        # Use only the [ACC] and [STR] tokens for the action prediction
        in_memory = sequence[:, self.act_tokens_pos, :]  # [B*S*cam, (H//P)^2+K, D] => [B*S*cam, t, D]; t = 2

        # Steering
        steering_in_memory = rearrange(in_memory[:, 0, :], '(B S cam) D -> B (S cam) D', B=B, S=S,
                                       cam=cam)  # [B, S*cam, D]
        steering_in_memory = torch.cat([self.final_steer_token, steering_in_memory],
                                       dim=1)  # [B, S*cam, D] => [B, S*cam+1, D]
        steering_in_memory = steering_in_memory + self.pos_embed_steering_tfx  # [B, S*cam+1, D]
        steering_in_memory = self.steering_tfx_encoder.forward(steering_in_memory)  # [B, S*cam+1, D]
        steering = reduce(steering_in_memory[:, 0], 'B D -> B 1', 'mean')  # [B, D] => [B, 1]

        # Acceleration
        accel_in_memory = rearrange(in_memory[:, 1, :], '(B S cam) D -> B (S cam) D', B=B, S=S,
                                    cam=cam)  # [B, S*cam, D]
        accel_in_memory = torch.cat([self.final_accel_token, accel_in_memory],
                                    dim=1)  # [B, S*cam, D] => [B, S*cam+1, D]
        accel_in_memory = accel_in_memory + self.pos_embed_accel_tfx  # [B, S*cam+1, D]
        accel_in_memory = self.accel_tfx_encoder.forward(accel_in_memory)  # [B, S*cam+1, D]
        acceleration = reduce(accel_in_memory[:, 0], 'B D -> B 1', 'mean')  # [B, D] => [B, 1]

        # Action prediction
        action_output = torch.stack((steering, acceleration), dim=2)  # [B, 1, t] = [B, 1, len(TARGETS)]

        return action_output

    def forward(self, s, s_d, s_s):
        """
        Arguments:
            s: images
            s_d: directions/commands (one-hot)rearrange
            s_s: speed (in m/s)

        """
        # Data dimensions
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]  # Batch size
        cam = len(g_conf.DATA_USED)  # Number of cameras/views

        # Encode the observations
        encoded_obs = self.encode_observations(s, s_d, s_s)  # [B*S*cam, (H//P)^2 + K, D]

        # Add positional encoding/embedding
        encoded_obs = self.pos_embed_camera_tfx(encoded_obs) if not g_conf.LEARNABLE_POS_EMBED else encoded_obs + self.pos_embed_camera_tfx  # [B*S*cam, H*W/P^2 + K, D]]

        # Pass on to the camera Transformer encoder
        in_memory = self.camera_tfx_encoder.forward(encoded_obs)  # [B*S*cam, (H//P)^2+K, D]

        # Get the action prediction
        action_output = self.action_prediction(in_memory)  # [B, 1, t] = [B, 1, len(TARGETS)]

        return action_output

    def forward_eval(self, s, s_d, s_s, attn_rollout: bool = False):
        """
        Args:
            s: Images
            s_d: Direction/command (one-hot)
            s_s: Speed (in m/s)
            attn_rollout: Whether to use Attention Rollout on the attention maps

        Returns:

        """
        # Data dimensions
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]  # Batch size
        cam = len(g_conf.DATA_USED)  # Number of cameras/views

        # Encode the observations
        encoded_obs = self.encode_observations(s, s_d, s_s)  # [B*S*cam, (H//P)^2 + K, D]

        # Add positional encoding/embedding
        encoded_obs = self.pos_embed_camera_tfx(encoded_obs) if not g_conf.LEARNABLE_POS_EMBED else encoded_obs + self.pos_embed_camera_tfx  # [B*S*cam, H*W/P^2 + K, D]]

        # Pass on to the camera Transformer encoder
        in_memory, attn_weights = self.tfx_encoder.forward_return_attn(encoded_obs)  # [B*S*cam, (H//P)^2+K, D], num_layers * [B*S*cam, (H//P)^2+K, (H//P)^2+K]

        # Get the action prediction
        action_output = self.action_prediction(in_memory)  # [B, 1, t] = [B, 1, len(TARGETS)]

        # Attention stuff
        if attn_rollout:
            attn_weights = utils.attn_rollout(attn_weights)

        if g_conf.CMD_SPD_TOKENS:
            # We'd like to analyze the attention weights of the [CMD] and [SPD] tokens
            attn_weights = attn_weights[-1][:, :self.act_tokens_pos[-1]+1, -self.tfx_num_patches:]  # [B*S*cam, t+2, (H//P)^2]
            # Normalize/make sure the sum is 1 w.r.t. the last dimension
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)  # [B*S*cam, t+2, (H//P)^2]
            attn_weights = attn_weights.unflatten(2, (self.tfx_num_patches_h, self.tfx_num_patches_w))  # [B*S*cam, t+2, H//P, W//P]
        else:
            # Return only the attention weights of the last layer for the [ACC] and [STR] tokens
            attn_weights = attn_weights[-1][:, self.act_tokens_pos, -self.tfx_num_patches:]  # [B*S*cam, t, (H//P)^2]
            # Normalize/make sure the sum is 1 w.r.t. the last dimension
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)  # [B*S*cam, t, (H//P)^2]
            attn_weights = attn_weights.unflatten(2, (self.tfx_num_patches_h, self.tfx_num_patches_w))  # [B*S*cam, t, H//P, W//P]

        return action_output, attn_weights

    @staticmethod
    def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
