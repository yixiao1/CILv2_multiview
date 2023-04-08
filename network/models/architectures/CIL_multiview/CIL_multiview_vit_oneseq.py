import torch
import torch.nn as nn
import importlib

from configs import g_conf
from network.models.building_blocks import FC
from network.models.building_blocks.PositionalEncoding import PositionalEncoding
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoder
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoderLayer

from einops import rearrange


class CIL_multiview_vit_oneseq(nn.Module):
    def __init__(self, params):
        super(CIL_multiview_vit_oneseq, self).__init__()
        self.params = params

        # Get ViT model characteristics (our new perception module)
        vit_module = importlib.import_module('network.models.building_blocks.vit')
        vit_module = getattr(vit_module, params['encoder_embedding']['perception']['res']['name'])
        self.encoder_embedding_perception = vit_module(pretrained=g_conf.IMAGENET_PRE_TRAINED)

        # Network pieces
        self.tfx_class_token = self.encoder_embedding_perception.class_token  # [1, 1, D], D is the hidden dimension
        self.tfx_conv_projection = self.encoder_embedding_perception.conv_proj
        self.tfx_encoder = self.encoder_embedding_perception.encoder

        # Get the vision transformer characteristics
        self.tfx_hidden_dim = self.encoder_embedding_perception.hidden_dim  # D
        self.tfx_patch_size = self.encoder_embedding_perception.patch_size  # P
        self.tfx_image_size = self.encoder_embedding_perception.image_size  # H, W

        # Token characteristics
        old_seq_length = self.encoder_embedding_perception.seq_length  # (H//P)^2 + 1
        old_patch_number = old_seq_length - 1  # (H//P)^2
        old_seq_length_1d = int(old_patch_number ** 0.5)  # H//P; keep same names as in official code

        self.tfx_patch_number = int(g_conf.ENCODER_INPUT_FRAMES_NUM) * len(g_conf.DATA_USED) * old_patch_number  # S*cam*(H//P)^2
        new_seq_shape = (int(g_conf.ENCODER_INPUT_FRAMES_NUM) * len(g_conf.DATA_USED) * old_seq_length_1d,
                         old_seq_length_1d)  # [S*cam*H//P, H//P]

        # Interpolate the positional encoding to the correct size
        old_pos_embedding_token = self.tfx_encoder.pos_embedding[:, :1, :]  # [1, 1, D]
        old_pos_embedding_img = self.tfx_encoder.pos_embedding[:, 1:, :]  # [1, (H//P)^2, D]
        old_pos_embedding_img = rearrange(old_pos_embedding_img, '1 (p1 p2) d -> 1 d p1 p2', p1=old_seq_length_1d)  # [1, D, H//P, H//P]

        # Create the positional embedding; will be used implicitly by the Encoder
        if g_conf.IMAGENET_PRE_TRAINED:
            # If pre-trained, we need to interpolate the positional embedding
            new_pos_embedding_img = nn.functional.interpolate(
                old_pos_embedding_img,  # Grab the old positional embedding, [1, D, H//P, H//P]
                size=new_seq_shape,  # expand it as we have concatenated the images, [S*cam*H//P, H//P]
                mode='bicubic',
                align_corners=True
            )  # [1, D, S*cam*H//P, H//P]
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

        join_dim = self.tfx_hidden_dim  # params['TxEncoder']['d_model']

        self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, self.tfx_hidden_dim)#  params['TxEncoder']['d_model'])
        self.speed = nn.Linear(1, self.tfx_hidden_dim)  # params['TxEncoder']['d_model'])

        self.action_output = FC(params={'neurons': [join_dim] +
                                            params['action_output']['fc']['neurons'] +
                                            [len(g_conf.TARGETS)],
                                 'dropouts': params['action_output']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        # We don't want to undo the pretrained weights of the ViT!
        for name, module in self.named_modules():
            if 'encoder' not in name:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.1)

        self.train()

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

        # Image info
        C, H, W = g_conf.IMAGE_SHAPE

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1) # [B, S, cam, 3, H, W]
        x = x.view(B*S*cam, C, H, W)  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        # First, patch and embed all the input images
        e_p = self.tfx_conv_projection(x)  # [B*S*cam, D, H//P, W//P]
        e_p = rearrange(e_p, '(batch S cam) D patches_H patches_W -> batch (S cam patches_H patches_W) D', 
                        S=S, cam=cam, D=self.tfx_hidden_dim)  # [B, S*cam*((H//P)^2), D]

        # Now the rest of forward
        e_p = torch.cat([self.tfx_class_token.expand(B, -1, -1), e_p], dim=1)  # [B, S*cam*(H//P)^2 + 1, D]

        # Embedding of command and speed
        e_d = self.command(d).unsqueeze(1)     # [B, 1, D]
        e_s = self.speed(s).unsqueeze(1)       # [B, 1, D]

        # Add the embeddings to the image embeddings (TODO: try different ways to do this)
        encoded_obs = e_p + e_d + e_s  # [B, S*cam*(H//P)^2 + 1, D]]

        # Pass on to the Transformer encoder
        in_memory = self.tfx_encoder(encoded_obs)  # [B, S*cam*(H//P)^2 + 1, D]]
        # Use only the [CLS] token for the action prediction 
        in_memory = in_memory[:, 0, :]  # [B, S*cam*(H//P)^2 + 1, D] => [B, D]

        # Action prediction
        action_output = self.action_output(in_memory).unsqueeze(1)  # (B, D) -> (B, 1, len(TARGETS))

        return action_output

    def forward_eval(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]  # Batch size; will change if using multiple GPUs
        cam = len(g_conf.DATA_USED)  # Number of cameras

        # Image info
        C, H, W = g_conf.IMAGE_SHAPE

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = x.view(B * S * cam, C, H, W)  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        # First, patch and embed all the input images
        e_p = self.tfx_conv_projection(x)  # [B*S*cam, D, H//P, W//P]
        e_p = rearrange(e_p, '(batch S cam) D patches_H patches_W -> batch (S cam patches_H patches_W) D',
                        S=S, cam=cam, D=self.tfx_hidden_dim)  # [B, S*cam*((H//P)^2), D]

        # Now the rest of forward
        e_p = torch.cat([self.tfx_class_token.expand(B, -1, -1), e_p], dim=1)  # [B, S*cam*(H//P)^2 + 1, D]

        # Embedding of command and speed
        e_d = self.command(d).unsqueeze(1)  # [B, 1, D]
        e_s = self.speed(s).unsqueeze(1)  # [B, 1, D]

        # Add the embeddings to the image embeddings (TODO: try different ways to do this)
        encoded_obs = e_p + e_d + e_s  # [B, S*cam*(H//P)^2 + 1, D]]

        # Pass on to the Transformer encoder
        in_memory = self.tfx_encoder(encoded_obs)  # [B, S*cam*(H//P)^2 + 1, D]]
        # Use only the [CLS] token for the action prediction
        in_memory = in_memory[:, 0, :]  # [B, S*cam*(H//P)^2 + 1, D] => [B, D]

        # Action prediction
        action_output = self.action_output(in_memory).unsqueeze(1)  # (B, D) -> (B, 1, len(TARGETS))

        return action_output  # TODO: return attn_weights?

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

