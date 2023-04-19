import importlib

import torch
import torch.nn as nn
from einops import rearrange

from configs import g_conf
from network.models.building_blocks.PositionalEncoding import PositionalEncoding


class CIL_vit(nn.Module):
    def __init__(self, params):
        super(CIL_vit, self).__init__()
        self.params = params

        # Get ViT model characteristics (our new perception module)
        vit_module = importlib.import_module('network.models.building_blocks.vit')
        vit_module = getattr(vit_module, params['encoder_embedding']['perception']['vit']['name'])
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

        self.tfx_seq_length = self.tfx_num_patches + 3  # K=3 for the additional tokens
        self.act_tokens_pos = [0, 1]  # Which tokens in the sequence to use for the action

        if g_conf.FREEZE_CLS_TOKEN:
            print('Freezing the [CLS] token...')
            self.tfx_class_token.requires_grad = False

        # Remove the [CLS] token if needed
        if g_conf.REMOVE_CLS_TOKEN:
            print('Removing the [CLS] token from the sequence...')
            self.tfx_seq_length -= 1  # K=2
            del self.tfx_class_token

        # Replace learned pos embedding with fixed sin/cos 2d embedding, used implicitly by the encoder
        self.setup_pos_embedding()

        # We don't want to undo the pretrained weights of the ViT!
        for name, module in self.named_modules():
            if 'encoder' not in name:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.1)

        self.train()

    def setup_pos_embedding(self):
        """ Setup the positional embedding of the network """
        if g_conf.LEARNABLE_POS_EMBED and not g_conf.IMAGENET_PRE_TRAINED:
            self.tfx_encoder.pos_embedding = nn.Parameter(
                torch.empty(1, self.tfx_seq_length, self.tfx_hidden_dim).normal_(std=0.02))  # From BERT
        elif g_conf.LEARNABLE_POS_EMBED and g_conf.IMAGENET_PRE_TRAINED:
            if self.tfx_seq_length == self.encoder_embedding_perception.seq_length:
                # Sequence length is the same as the pretrained one, so we can use the pretrained one
                pass
            else:
                # TODO: Is there a way to interpolate from [1, S, hidden_dim] to [1, new_S, hidden_dim]?
                print('Warning: current sequence length is different from the pretrained one. Starting '
                      'Positional Encoding from scratch...')
                self.tfx_encoder.pos_embedding = nn.Parameter(
                    torch.empty(1, self.tfx_seq_length, self.tfx_hidden_dim).normal_(std=0.02))

                print('Warning: current sequence length is different from the pretrained one. Starting from scratch...')
                self.tfx_encoder.pos_embedding = nn.Parameter(
                    torch.empty(1, self.tfx_seq_length, self.tfx_hidden_dim).normal_(std=0.02))

        else:
            del self.tfx_encoder.pos_embedding  # A nn.Parameter, so it most go

            self.tfx_encoder.pos_embedding = PositionalEncoding(
                d_model=self.tfx_hidden_dim,
                max_len=self.tfx_seq_length).pe.cuda()  # Super hacky; TODO: fix this

    def forward(self, s, s_d, s_s):
        """
        Arguments:
            s: images
            s_d: directions/commands (one-hot)rearrange
            s_s: speed (in m/s)

        """
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]     # Batch size
        cam = len(g_conf.DATA_USED)  # Number of cameras/views

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1) # [B, S, cam, 3, H, W]
        x = x.view(B*S*cam, g_conf.IMAGE_SHAPE[0], g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2])  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        # First, patch and embed the input images
        e_p = self.tfx_conv_projection(x)  # [B*S*cam, D, H/P, W/P]; usually H = W = 224
        e_p = rearrange(e_p, '(B S cam) D (patches_h) (patches_w) -> (B S cam) (patches_h patches_w) D',
                        B=B, cam=cam)  # [B*S*cam, H*W/P^2, D]

        # Setup the first tokens in the sequence
        n = e_p.shape[0]  # B*S*cam
        if g_conf.REMOVE_CLS_TOKEN:
            first_tokens = [self.tfx_accel_token.expand(n, -1, -1),
                            self.tfx_steer_token.expand(n, -1, -1)]
        else:
            first_tokens = [self.tfx_accel_token.expand(n, -1, -1),
                            self.tfx_steer_token.expand(n, -1, -1),
                            self.tfx_class_token.expand(n, -1, -1)]

        # Concatenate the first tokens to the image embeddings
        e_p = torch.cat([*first_tokens, e_p], dim=1)  # [B*S*cam, (H/P)(W/P) + K, D]]), K = 3 if w/CLS token else 2

        # Embedding of command and speed
        e_d = self.command(d).unsqueeze(1)     # [B, 1, D]
        e_s = self.speed(s).unsqueeze(1)       # [B, 1, D]

        # Add the embeddings to the image embeddings
        if g_conf.CMD_SPD_TOKENS:
            e_d = e_d.repeat(S * cam, 1, 1)  # [B*S*cam, 1, D]
            e_s = e_s.repeat(S * cam, 1, 1)  # [B*S*cam, 1, D]
            self.tfx_seq_length += 2
            self.act_tokens_pos = [_ + 2 for _ in self.act_tokens_pos]
            self.setup_pos_embedding()  # New sequence length, so we need to update the positional embedding
            encoded_obs = torch.cat([e_p, e_d, e_s], dim=1)  # [B*S*cam, (H//P)^2 + 5, D]; K = 5 w/CLS token else 4
        else:
            e_p = e_p.reshape(B, -1, self.tfx_hidden_dim)  # [B, S*cam*((H//P)^2 + 3), D]
            encoded_obs = e_p + e_d + e_s  # [B, S*cam*((H//P)^2 + K), D]; K = 3 if not removing CLS token else 2

        encoded_obs = encoded_obs.reshape(-1, self.tfx_seq_length, self.tfx_hidden_dim)  # [B*S*cam, (H//P)^2 + K, D]

        # Pass on to the Transformer encoder
        in_memory = self.tfx_encoder(encoded_obs)  # [B*S*cam, (H//P)^2+K, D]
        # Use only the [ACC] and [STR] tokens for the action prediction
        in_memory = in_memory[:, self.act_tokens_pos, :]  # [B*S*cam, (H//P)^2+K, D] => [B*S*cam, t, D]; t = 2
        in_memory = rearrange(in_memory, '(B S cam) t D -> B (S cam D) t', B=B, S=S)  # [B*S*cam, t, D] => [B, S*cam*D]

        # Action prediction
        action_output = torch.mean(in_memory, dim=1, keepdim=True)  # [B, 1, t] = [B, 1, len(TARGETS)]

        return action_output

    def forward_eval(self, s, s_d, s_s):
        return self.forward(s, s_d, s_s)  # TODO: return attn_weights?

    @staticmethod
    def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
