import torch
import torch.nn as nn
import importlib

from configs import g_conf
from network.models.building_blocks import FC

from einops import rearrange


class CIL_multiview_vit_multitokens(nn.Module):
    def __init__(self, params):
        super(CIL_multiview_vit_multitokens, self).__init__()
        self.params = params

        # Get ViT model characteristics (our new perception module)
        vit_module = importlib.import_module('network.models.building_blocks.vit')
        vit_module = getattr(vit_module, params['encoder_embedding']['perception']['vit']['name'])
        self.encoder_embedding_perception = vit_module(pretrained=g_conf.IMAGENET_PRE_TRAINED)

        # Get the vision transformer characteristics
        self.tfx_hidden_dim = self.encoder_embedding_perception.hidden_dim  # D
        self.tfx_seq_length = self.encoder_embedding_perception.seq_length  # (H//P)^2 + 1  (1 for the class token)
        self.tfx_patch_size = self.encoder_embedding_perception.patch_size  # P
        self.tfx_image_size = self.encoder_embedding_perception.image_size  # H, W
        # Network pieces
        self.tfx_class_token = self.encoder_embedding_perception.class_token  # Idea is that this is pretrained
        self.tfx_conv_projection = self.encoder_embedding_perception.conv_proj  # Ibidem
        self.tfx_encoder = self.encoder_embedding_perception.encoder
        self.tfx_pos_embedding = self.tfx_encoder.pos_embedding  # Used implicitly by the encoder above

        self.join_dim = self.tfx_hidden_dim  # params['TxEncoder']['d_model']

        self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, self.tfx_hidden_dim)  # params['TxEncoder']['d_model'])
        self.speed = nn.Linear(1, self.tfx_hidden_dim)  # params['TxEncoder']['d_model'])

        self.action_output = FC(params={'neurons': [self.join_dim] +
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
        B = s_d[0].shape[0]     # Batch size

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1) # [B, S, cam, 3, H, W]
        x = x.view(B*S*len(g_conf.DATA_USED), g_conf.IMAGE_SHAPE[0], g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2])  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        # First, patch and embed the input images
        e_p = self.tfx_conv_projection(x)  # [B*S*cam, D, H//P, W//P]
        e_p = e_p.reshape(-1, self.tfx_hidden_dim, self.tfx_seq_length - 1)  # [B*S*cam, D, (H//P)^2]
        e_p = e_p.permute(0, 2, 1)  # [B*S*cam, (H//P)^2, D]

        # Now the rest of forward
        n = e_p.shape[0]
        e_p = torch.cat([self.tfx_class_token.expand(n, -1, -1), e_p], dim=1)  # [B*S*cam, (H//P)^2 + 1, D]])
        e_p = e_p.reshape(B, -1, self.tfx_hidden_dim)  # [B, S*cam*((H//P)^2 + 1), D]

        # Embedding of command and speed
        e_d = self.command(d).unsqueeze(1)     # [B, 1, D]
        e_s = self.speed(s).unsqueeze(1)       # [B, 1, D]

        # Add the embeddings to the image embeddings (TODO: try different ways to do this)
        encoded_obs = torch.cat([e_p, e_d, e_s], dim=1)  # [B, S*cam*((H//P)^2 + 3), D]
        encoded_obs = encoded_obs.reshape(-1, self.tfx_seq_length + 2, self.tfx_hidden_dim)  # [B*S*cam, (H//P)^2 + 3, D]

        # TODO: do this manually to use our own positional encoding
        # Pass on to the Transformer encoder
        in_memory = self.tfx_encoder(encoded_obs)  # [B*S*cam, (H//P)^2+1, D]
        # TODO: Use the [CMD] token for the action prediction
        in_memory = in_memory[:, 0].reshape(B, -1, self.tfx_hidden_dim)  # [B*S*cam, (H//P)^2+1, D] => [B*S*cam, D] => [B, S*cam, D]
        in_memory = torch.mean(in_memory, dim=1)  # [B, D]

        # Action prediction
        action_output = self.action_output(in_memory).unsqueeze(1)  # (B, D) -> (B, 1, len(TARGETS))

        return action_output

    def forward_eval(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d[0].shape[0]

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = x.view(B * S * len(g_conf.DATA_USED), g_conf.IMAGE_SHAPE[0], g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2])  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        # First, patch and embed the input images
        e_p = self.tfx_conv_projection(x)  # [B*S*cam, D, H//P, W//P]
        e_p = e_p.reshape(-1, self.tfx_hidden_dim, self.tfx_seq_length - 1)  # [B*S*cam, D, (H//P)^2]
        e_p = e_p.permute(0, 2, 1)  # [B*S*cam, (H//P)^2, D]

        # Now the rest of forward
        n = e_p.shape[0]
        e_p = torch.cat([self.tfx_class_token.expand(n, -1, -1), e_p], dim=1)  # [B*S*cam, (H//P)^2 + 1, D]])
        e_p = e_p.reshape(B, -1, self.tfx_hidden_dim)  # [B, S*cam*((H//P)^2 + 1), D]

        e_d = self.command(d).unsqueeze(1)  # [B, 1, 512]
        e_s = self.speed(s).unsqueeze(1)  # [B, 1, 512]

        encoded_obs = e_p + e_d + e_s  # [B, S*cam*((H//P)^2 + 1), D]
        encoded_obs = encoded_obs + e_d + e_s   # [B, S*cam*h*w, 512]

        # Add the embeddings to the image embeddings (TODO: try different ways to do this)
        encoded_obs = e_p + e_d + e_s  # [B, S*cam*((H//P)^2 + 1), D]
        encoded_obs = encoded_obs.reshape(-1, self.tfx_seq_length, self.tfx_hidden_dim)  # [B*S*cam, (H//P)^2 + 1, D]

        # Pass on to the Transformer encoder
        in_memory = self.tfx_encoder(encoded_obs)  # [B*S*cam, (H//P)^2+1, D]
        # Use only the [CLS] token for the action prediction
        in_memory = in_memory[:, 0].reshape(B, -1, self.tfx_hidden_dim)  # [B*S*cam, (H//P)^2+1, D] => [B*S*cam, D] => [B, S*cam, D]
        in_memory = torch.mean(in_memory, dim=1)  # [B, D]

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

