import torch
import torch.nn as nn
import importlib

from configs import g_conf
from network.models.building_blocks.PositionalEncoding import PositionalEncoding

from einops import rearrange
from typing import Tuple


class CIL_multiview_vit_oneseq(nn.Module):
    def __init__(self, params):
        super(CIL_multiview_vit_oneseq, self).__init__()
        self.params = params

        # Get ViT model characteristics (our new perception module)
        vit_module = importlib.import_module('network.models.building_blocks.vit')
        try:
            module_name = self.params['encoder_embedding']['perception']['vit']['name']
        except KeyError:
            module_name = self.params['encoder_embedding']['perception']['res']['name']
        vit_module = getattr(vit_module, module_name)
        self.encoder_embedding_perception = vit_module(pretrained=g_conf.IMAGENET_PRE_TRAINED)

        self.image_channels, self.image_height, self.image_width = g_conf.IMAGE_SHAPE

        # Get the vision transformer characteristics
        self.tfx_hidden_dim = self.encoder_embedding_perception.hidden_dim  # D
        self.tfx_patch_size = self.encoder_embedding_perception.patch_size  # P
        self.tfx_num_patches_h = self.image_height // self.tfx_patch_size  # H/P
        self.tfx_num_patches_w = self.image_width // self.tfx_patch_size  # W/P
        self.tfx_num_patches = self.tfx_num_patches_h * self.tfx_num_patches_w  # (H/P)*(W/P)

        # Network pieces
        self.tfx_conv_projection = self.encoder_embedding_perception.conv_proj
        self.tfx_encoder = self.encoder_embedding_perception.encoder

        self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, self.tfx_hidden_dim)
        self.speed = nn.Linear(1, self.tfx_hidden_dim)

        # Additional tokens
        self.tfx_class_token = self.encoder_embedding_perception.class_token
        if g_conf.PRETRAINED_ACC_STR_TOKENS:
            # Start from the [CLS] token of the pretrained model
            print('Initializing the acceleration ([ACC]) and steering ([STR]) tokens with the [CLS] token...')
            self.tfx_accel_token = nn.Parameter(self.tfx_class_token.detach().clone())
            self.tfx_steer_token = nn.Parameter(self.tfx_class_token.detach().clone())
        else:
            # Start from scratch
            print('Randomly initializing the acceleration ([ACC]) and steering ([STR]) tokens...')
            self.tfx_accel_token = nn.Parameter(torch.empty(1, 1, self.tfx_hidden_dim).normal_(std=0.02))
            self.tfx_steer_token = nn.Parameter(torch.empty(1, 1, self.tfx_hidden_dim).normal_(std=0.02))

        self.tfx_seq_length = int(g_conf.ENCODER_INPUT_FRAMES_NUM) * len(g_conf.DATA_USED) * self.tfx_num_patches + 3  # K=3 for the additional tokens
        self.act_tokens_pos = [0, 1]  # Which tokens in the sequence to use for the action

        # Remove the [CLS] token if needed
        if g_conf.REMOVE_CLS_TOKEN:
            print('Removing the [CLS] token from the sequence...')
            self.tfx_seq_length -= 1  # K=2
            del self.tfx_class_token

        # Add the command and speed as tokens to the sequence
        if g_conf.CMD_SPD_TOKENS:
            self.tfx_seq_length += 2
            self.act_tokens_pos = [_ + 2 for _ in self.act_tokens_pos]

        # Positional Encoding (fixed for now)
        del self.tfx_encoder.pos_embedding  # A nn.Parameter, so it most go
        self.pos_embedding = PositionalEncoding(d_model=self.tfx_hidden_dim, max_len=self.tfx_seq_length)

        # We don't want to undo the pretrained weights of the ViT!
        for name, module in self.named_modules():
            if 'encoder' not in name:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.1)

        self.train()

    def interpolate_pos_encoding(self, new_seq_shape: Tuple[int, int]) -> None:
        # Interpolate the positional encoding to the correct size
        old_pos_embedding_token = self.tfx_encoder.pos_embedding[:, :1, :]
        old_pos_embedding_img = self.tfx_encoder.pos_embedding[:, 1:, :]
        old_pos_embedding_img = rearrange(old_pos_embedding_img, '1 (p1 p2) d -> 1 d p1 p2', p1=self.tfx_patch_size)

        # Create the positional embedding; will be used implicitly by the Encoder
        if g_conf.IMAGENET_PRE_TRAINED:
            # If pre-trained, we need to interpolate the positional embedding
            new_pos_embedding_img = nn.functional.interpolate(
                old_pos_embedding_img,  # Grab the old positional embedding, [1, D, H//P, H//P]
                size=new_seq_shape,  # expand it as we have concatenated the images, [S*cam*H//P, H//P]
                mode='bicubic',
                align_corners=True)  # [1, D, S*cam*H//P, H//P]

            new_pos_embedding_img = rearrange(new_pos_embedding_img, '1 D (S cam p1) p2 -> 1 (S cam p1 p2) D',
                                              D=self.tfx_hidden_dim, S=int(g_conf.ENCODER_INPUT_FRAMES_NUM),
                                              cam=len(g_conf.DATA_USED))  # [1, S*cam*(H//P)^2, D]
            self.tfx_encoder.pos_embedding = nn.Parameter(
                torch.cat([old_pos_embedding_token, new_pos_embedding_img], dim=1))  # [1, S*cam*(H//P)^2 + 1, D]
        else:
            # If we are not using a pre-trained model, we need to create the positional embedding from scratch
            # (as interpolating the old one would not make sense); we use the same method as in the official code
            self.tfx_encoder.pos_embedding = nn.Parameter(
                torch.empty(1, self.tfx_patch_number + 1, self.tfx_hidden_dim).normal_(std=0.02)
            )

    def encode_observations(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]  # Batch size
        cam = len(g_conf.DATA_USED)  # Number of cameras/views

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = rearrange(x, 'B S cam C H W -> (B S cam) C H W')  # [B*S*cam, C=3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        # First, patch and embed the input images
        e_p = self.tfx_conv_projection(x)  # [B*S*cam, D, H/P, W/P]; usually H = W = 224
        e_p = rearrange(e_p, '(B S cam) D patches_H patches_W -> B (S cam patches_H patches_W) D',
                        B=B, cam=cam, D=self.tfx_hidden_dim)  # [B, S*cam*H*W/P^2, D]

        # Setup the first tokens in the sequence
        n = e_p.shape[0]  # B
        first_tokens = [self.tfx_steer_token.expand(n, -1, -1),
                        self.tfx_accel_token.expand(n, -1, -1)][::-2 * g_conf.OLD_TOKEN_ORDER + 1]
        if not g_conf.REMOVE_CLS_TOKEN:
            first_tokens = [*first_tokens, self.tfx_class_token.expand(n, -1, -1)]

        # Concatenate the first tokens to the image embeddings
        e_p = torch.cat([*first_tokens, e_p], dim=1)  # [B, S*cam*H*W/P^2 + K, D]]), K = 3 if w/CLS token else 2

        # Embedding of command and speed
        e_d = self.command(d).unsqueeze(1)  # [B, 1, D]
        e_s = self.speed(s).unsqueeze(1)  # [B, 1, D]

        # Add the embeddings to the image embeddings
        if g_conf.CMD_SPD_TOKENS:
            encoded_obs = torch.cat([e_p, e_d, e_s], dim=1)  # [B, S*cam*H*W/P^2 + K, D]; K = 5 w/CLS token else 4
        else:
            encoded_obs = e_p + e_d + e_s  # [B, S*cam*H*W/P^2 + K), D]; K = 3 w/CLS token else 2

        return encoded_obs

    def forward(self, s, s_d, s_s):
        """
        Arguments:
            s: images
            s_d: directions/commands (one-hot)
            s_s: speed (in m/s)

        """
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]     # Batch size; will change if using multiple GPUs
        cam = len(g_conf.DATA_USED)  # Number of cameras

        encoded_obs = self.encode_observations(s, s_d, s_s)  # [B, S*cam*H*W/P^2 + K, D]

        # Add positional encoding
        encoded_obs = self.pos_embedding(encoded_obs)  # [B, S*cam*H*W/P^2 + K, D]]

        # Pass on to the Transformer encoder
        in_memory = self.tfx_encoder(encoded_obs, pre_pos_embed=True)  # [B, S*cam*H*W/P^2 + K, D]

        # Action prediction: use only the action tokens for the prediction
        in_memory = in_memory[:, self.act_tokens_pos]  # [B, S*cam*H*W/P^2 + K, D] => [B, t, D]; t = 2
        in_memory = rearrange(in_memory, 'B t D -> B D t')  # [B, t, D] => [B, D, t]

        # Action prediction
        action_output = torch.mean(in_memory, dim=1, keepdim=True)  # [B, D, t] -> (B, 1, len(TARGETS))

        return action_output

    def forward_eval(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]  # Batch size; will change if using multiple GPUs
        cam = len(g_conf.DATA_USED)  # Number of cameras

        encoded_obs = self.encode_observations(s, s_d, s_s)  # [B, S*cam*H*W/P^2 + K, D]

        # Add positional encoding
        encoded_obs = self.pos_embedding(encoded_obs)  # [B, S*cam*H*W/P^2 + K, D]]

        # Pass on to the Transformer encoder
        # [B, S*cam*H*W/P^2 + K, D], num_layers * [B, S*cam*H*W/P^2 + K, S*cam*H*W/P^2 + K]
        in_memory, attn_weights = self.tfx_encoder.forward_return_attn(encoded_obs, pre_pos_embed=True)

        # Action prediction: use only the action tokens for the prediction
        in_memory = in_memory[:, self.act_tokens_pos]  # [B, S*cam*H*W/P^2 + K, D] => [B, t, D]; t = 2
        in_memory = rearrange(in_memory, 'B t D -> B D t')  # [B, t, D] => [B, D, t]

        # Action prediction
        action_output = torch.mean(in_memory, dim=1, keepdim=True)  # [B, D, t] -> (B, 1, len(TARGETS))

        return action_output

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

