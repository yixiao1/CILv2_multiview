import torch
import torch.nn as nn
import torchvision
import importlib

from configs import g_conf
from _utils import utils
from network.models.building_blocks.PositionalEncoding import PositionalEncoding
from network.models.building_blocks.fc import FC

from einops import rearrange, reduce
from typing import Tuple, List


class CIL_multiview_vit_oneseq(nn.Module):
    def __init__(self, params, rank: int = 0):
        super(CIL_multiview_vit_oneseq, self).__init__()
        self.params = params

        # Get ViT model characteristics (our new perception module)
        vit_module = importlib.import_module('network.models.building_blocks.vit')
        try:
            module_name = self.params['encoder_embedding']['perception']['vit']['name']
        except KeyError:
            module_name = self.params['encoder_embedding']['perception']['res']['name']
        vit_module = getattr(vit_module, module_name)
        encoder_embedding_perception = vit_module(pretrained=g_conf.IMAGENET_PRE_TRAINED)

        self.image_channels, self.image_height, self.image_width = g_conf.IMAGE_SHAPE

        # Get the vision transformer characteristics
        self.tfx_hidden_dim = encoder_embedding_perception.hidden_dim  # D
        self.tfx_patch_size = encoder_embedding_perception.patch_size  # P
        self.tfx_num_patches_h = self.image_height // self.tfx_patch_size  # H/P
        self.tfx_num_patches_w = self.image_width // self.tfx_patch_size  # W/P
        self.tfx_num_patches = self.tfx_num_patches_h * self.tfx_num_patches_w  # (H/P)*(W/P), per image

        # Network pieces
        self.tfx_conv_projection = encoder_embedding_perception.conv_proj
        self.tfx_encoder = encoder_embedding_perception.encoder

        # Select the number of layers to use
        num_layers = self.params['encoder_embedding']['perception']['vit'].get('num_layers', None)
        if num_layers is not None and (num_layers != 'all' or num_layers != len(self.tfx_encoder.layers)):
            # Sanity check
            num_layers = int(num_layers)
            num_layers = max(min(num_layers, len(self.tfx_encoder.layers)), 1)
            self.tfx_encoder.num_layers = num_layers
            if rank == 0:
                print(f'Using the first {num_layers} layers of the ViT model (originally {len(self.tfx_encoder.layers)})...')
            layers = []
            for i in range(int(num_layers)):
                layers.append(self.tfx_encoder.layers[i])
            self.tfx_encoder.layers = nn.Sequential(*layers)

        freeze_layers = self.params['encoder_embedding']['perception']['vit'].get('freeze_layers', None)
        if freeze_layers is not None:
            # Sanity check
            if isinstance(freeze_layers, (str, int)):
                freeze_layers = list(range(int(freeze_layers)))
            elif isinstance(freeze_layers, list):
                freeze_layers = [int(x) for x in freeze_layers]
            freeze_layers = [x for x in freeze_layers if len(self.tfx_encoder.layers) > x >= 0]
            if len(freeze_layers) > 0:
                if rank == 0:
                    print(f'Freezing {len(freeze_layers)} layers of the Encoder (from total: {len(self.tfx_encoder.layers)})...')
                for i in freeze_layers:
                    for param in self.tfx_encoder.layers[i].parameters():
                        param.requires_grad = False

        if not g_conf.NEW_COMMAND_SPEED_FC:
            self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, self.tfx_hidden_dim)
            self.speed = nn.Linear(1, self.tfx_hidden_dim)
        else:
            self.command = FC(params={'neurons': [g_conf.DATA_COMMAND_CLASS_NUM, self.tfx_hidden_dim],
                                      'dropouts': self.params['command']['fc']['dropouts'],
                                      'end_layer': False},
                              norm=g_conf.FC_LAYER_NORM, activate=False)  # No sense in using ReLU here
            self.speed = FC(params={'neurons': [1, self.tfx_hidden_dim],
                                    'dropouts': self.params['speed']['fc']['dropouts'],
                                    'end_layer': False},
                            norm=g_conf.FC_LAYER_NORM, activate=False)  # No sense in using ReLU here

        if g_conf.NO_ACT_TOKENS:
            # Don't use any extra tokens in the sequence, so skip this
            self.tfx_seq_length = 0
        else:
            # Additional tokens
            self.tfx_class_token = encoder_embedding_perception.class_token

            if g_conf.ONE_ACTION_TOKEN:
                self.tfx_seq_length = 2  # Including the CLS token
                if g_conf.PRETRAINED_ACT_TOKENS or g_conf.PRETRAINED_ACC_STR_TOKENS:
                    # Start from the [CLS] token of the pretrained model
                    print('Initializing the action ([ACT]) token with the [CLS] token...') if rank == 0 else None
                    self.tfx_action_token = nn.Parameter(self.tfx_class_token.detach().clone())
                else:
                    # Start from scratch
                    print('Randomly initializing the action ([ACT]) token...') if rank == 0 else None
                    self.tfx_action_token = nn.Parameter(torch.empty(1, 1, self.tfx_hidden_dim).normal_(std=0.02))

            else:
                self.tfx_seq_length = 3  # Including the CLS token
                if g_conf.PRETRAINED_ACT_TOKENS or g_conf.PRETRAINED_ACC_STR_TOKENS:
                    # Start from the [CLS] token of the pretrained model
                    if rank == 0:
                        print('Initializing the acceleration ([ACC]) and steering ([STR]) tokens with the [CLS] token...')
                    self.tfx_accel_token = nn.Parameter(self.tfx_class_token.detach().clone())
                    self.tfx_steer_token = nn.Parameter(self.tfx_class_token.detach().clone())
                else:
                    # Start from scratch
                    if rank == 0:
                        print('Randomly initializing the acceleration ([ACC]) and steering ([STR]) tokens...')
                    self.tfx_accel_token = nn.Parameter(torch.empty(1, 1, self.tfx_hidden_dim).normal_(std=0.02))
                    self.tfx_steer_token = nn.Parameter(torch.empty(1, 1, self.tfx_hidden_dim).normal_(std=0.02))

            self.act_tokens_pos = [0, 1] if not g_conf.ONE_ACTION_TOKEN else [0]  # Tokens to use for the action

            # Remove the [CLS] token (default)
            if g_conf.REMOVE_CLS_TOKEN:
                self.tfx_seq_length -= 1  # K=2
                del self.tfx_class_token

            # Add the command and speed as tokens to the sequence
            if g_conf.CMD_SPD_TOKENS:
                if rank == 0:
                    print('Adding the command and speed tokens to the sequence...')
                self.tfx_seq_length += 2
                self.act_tokens_pos = [_ + 2 for _ in self.act_tokens_pos]

        # Sequence length
        self.tfx_seq_length += int(g_conf.ENCODER_INPUT_FRAMES_NUM) * len(g_conf.DATA_USED) * self.tfx_num_patches

        # Mask the diagonal with -inf
        if g_conf.MASK_DIAGONAL_ATTMAP:
            tfx_mask = torch.zeros((self.tfx_seq_length, self.tfx_seq_length)).fill_diagonal_(float('-inf'))
            self.tfx_mask = nn.Parameter(tfx_mask, requires_grad=False)
        else:
            self.tfx_mask = None

        if g_conf.LEARNABLE_POS_EMBED and not g_conf.IMAGENET_PRE_TRAINED:
            self.tfx_encoder.pos_embedding = nn.Parameter(
                torch.empty(1, self.tfx_seq_length, self.tfx_hidden_dim).normal_(std=0.02))  # From BERT
        elif g_conf.LEARNABLE_POS_EMBED and g_conf.IMAGENET_PRE_TRAINED:
            if self.tfx_seq_length == encoder_embedding_perception.seq_length:
                # Sequence length is the same as the pretrained one, so we can use the pretrained one
                pass
            else:
                # TODO: Is there a way to interpolate from [1, S, hidden_dim] to [1, new_S, hidden_dim]?
                if rank == 0:
                    print('Warning: current sequence length is different from the pretrained one. Starting '
                          'Positional Encoding from scratch...')
                self.tfx_encoder.pos_embedding = nn.Parameter(
                    torch.empty(1, self.tfx_seq_length, self.tfx_hidden_dim).normal_(std=0.02))

        else:
            del self.tfx_encoder.pos_embedding  # A nn.Parameter, so it most go

            self.pos_embedding = PositionalEncoding(d_model=self.tfx_hidden_dim, max_len=self.tfx_seq_length)

        if g_conf.SENSOR_EMBED:
            if rank == 0:
                print('Using a sensor embedding...')
            self.sensor_embedding = nn.Parameter(torch.zeros(1, self.tfx_seq_length, self.tfx_hidden_dim))

        if 'action_output' in self.params:
            if self.params['action_output']['type'] == '2mlp':
                self.steer_output = FC(params={
                    'neurons': [self.tfx_hidden_dim] + self.params['action_output']['fc']['neurons'] + [1],
                    'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                    'end_layer': True},
                    norm=g_conf.FC_LAYER_NORM, activate=True)  # Uses ReLU by default and on by default
                self.accel_output = FC(params={
                    'neurons': [self.tfx_hidden_dim] + self.params['action_output']['fc']['neurons'] + [1],
                    'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                    'end_layer': True},
                    norm=g_conf.FC_LAYER_NORM, activate=True)  # Uses ReLU by default and on by default
            elif self.params['action_output']['type'] == '1mlp':
                join_dim = self.tfx_hidden_dim * len(g_conf.TARGETS) if not g_conf.ONE_ACTION_TOKEN else self.tfx_hidden_dim
                self.action_output = FC(params={
                    'neurons': [join_dim] + self.params['action_output']['fc']['neurons'] + [len(g_conf.TARGETS)],
                    'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                    'end_layer': True},
                    norm=g_conf.FC_LAYER_NORM, activate=True)  # Uses ReLU by default and on by default
            elif self.params['action_output']['type'] in ['baseline1_patches2act', 'baseline3_patches2act_gap_avg', 'baseline4_patches2act_gap_diff']:
                self.action_output = FC(params={
                    'neurons': [self.tfx_hidden_dim] + self.params['action_output']['fc']['neurons'] + [len(g_conf.TARGETS)],
                    'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                    'end_layer': True},
                    norm=g_conf.FC_LAYER_NORM, activate=True)  # Uses ReLU by default and on by default
            else:
                # It's a GAP, so do nothing
                pass

        # Action ratio between pure tokens and patches
        if g_conf.LEARNABLE_ACTION_RATIO:
            # Start balanced
            self.action_ratio = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        else:
            self.action_ratio = 0.5

        # Extra regularization: predict the input speed and command
        if g_conf.CMD_SPD_TOKENS and g_conf.PREDICT_CMD_SPD:
            print('Adding the speed and command prediction...') if rank == 0 else None
            self.speed_output = FC(params={
                'neurons': [self.tfx_hidden_dim] + self.params['speed_output']['fc']['neurons'] + [1],
                'dropouts': self.params['speed_output']['fc']['dropouts'] + [0.0],
                'end_layer': True},
                norm=g_conf.FC_LAYER_NORM, activate=True)
            self.command_output = FC(params={
                'neurons': [self.tfx_hidden_dim] + self.params['command_output']['fc']['neurons'] + [g_conf.DATA_COMMAND_CLASS_NUM],
                'dropouts': self.params['command_output']['fc']['dropouts'] + [0.0],
                'end_layer': True},
                norm=g_conf.FC_LAYER_NORM, activate=True)

        # Legacy
        if g_conf.SAVE_FULL_STATE:
            self.encoder_embedding_perception = encoder_embedding_perception

        # We don't want to undo the pretrained weights of the ViT!
        for name, module in self.named_modules():
            if 'encoder' not in name:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.1)

        self.train()

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
        if not g_conf.NO_ACT_TOKENS:
            n = e_p.shape[0]  # B
            if g_conf.ONE_ACTION_TOKEN:
                first_tokens = [self.tfx_action_token.expand(n, -1, -1)]
            else:
                first_tokens = [self.tfx_steer_token.expand(n, -1, -1),
                                self.tfx_accel_token.expand(n, -1, -1)][::-2 * g_conf.OLD_TOKEN_ORDER + 1]
            if not g_conf.REMOVE_CLS_TOKEN:
                first_tokens = [*first_tokens, self.tfx_class_token.expand(n, -1, -1)]
        else:
            first_tokens = []

        # Concatenate the first tokens to the image embeddings
        e_p = torch.cat([*first_tokens, e_p], dim=1)  # [B, S*cam*H*W/P^2 + K, D]]), K = 3 if w/CLS token else 2

        # Embedding of command and speed
        e_d = self.command(d).unsqueeze(1)  # [B, 1, D]
        e_s = self.speed(s).unsqueeze(1)  # [B, 1, D]

        # Add the embeddings to the image embeddings
        if g_conf.CMD_SPD_TOKENS and not g_conf.NO_ACT_TOKENS:
            encoded_obs = torch.cat([e_d, e_s, e_p], dim=1)  # [B, S*cam*H*W/P^2 + K, D]; K = 5 w/CLS token else 4
        else:
            encoded_obs = e_p + e_d + e_s  # [B, S*cam*H*W/P^2 + K), D]; K = 3 w/CLS token else 2

        return encoded_obs

    def action_prediction(self, sequence: torch.Tensor, cam: int) -> torch.Tensor:
        """Pass in the sequence of the ViT (of shape [B, seq_length, hidden_dim]) and predict the action"""
        # Only use the action tokens [ACT] for the prediction
        if not g_conf.NO_ACT_TOKENS:
            sequence_act = sequence[:, self.act_tokens_pos]  # [B, seq_length, hidden_dim] => [B, t, D];, t = len(g_conf.TARGETS)
        if 'action_output' in self.params:
            if self.params['action_output']['type'] == 'baseline1_patches2act':
                # Get the patch representation at the final layer
                patches = reduce(sequence[:, -cam * self.tfx_num_patches:], 'B N D -> B D', 'mean')  # [B, N, D] => [B, D]
                # Pass the patch representation through an MLP
                action_output = self.action_output(patches).unsqueeze(1)  # [B, D] => [B, 1, t]
            elif self.params['action_output']['type'] == 'baseline2_gap' and not g_conf.ONE_ACTION_TOKEN:
                # Average pooling of the ACT tokens (one per target)
                action_output = reduce(sequence_act, 'B t D -> B 1 t', 'mean')  # [B, t, D] => [B, 1, t]
            elif self.params['action_output']['type'] == 'baseline3_patches2act_gap_avg' and not g_conf.ONE_ACTION_TOKEN:
                # Get the patch representation at the final layer
                patches = reduce(sequence[:, -cam * self.tfx_num_patches:], 'B N D -> B D', 'mean')  # [B, N, D] => [B, D]
                # Pass the patch representation through an MLP
                action_output_patches = self.action_output(patches).unsqueeze(1)  # [B, D] => [B, 1, t]
                # Average pooling of the ACT tokens (one per target)
                action_output_tokens = reduce(sequence_act, 'B t D -> B 1 t', 'mean')  # [B, t, D] => [B, 1, t]
                # Final action will be the average of both of these
                action_output = self.action_ratio * action_output_patches + (1 - self.action_ratio) * action_output_tokens
            elif self.params['action_output']['type'] == 'baseline4_patches2act_gap_diff' and not g_conf.ONE_ACTION_TOKEN:
                # Get the patch representation at the final layer
                patches = reduce(sequence[:, -cam * self.tfx_num_patches:], 'B N D -> B D', 'mean')  # [B, N, D] => [B, D]
                # Pass the patch representation through an MLP
                action_output_patches = self.action_output(patches).unsqueeze(1)  # [B, D] => [B, 1, t]
                # Average pooling of the ACT tokens (one per target)
                action_output_tokens = reduce(sequence_act, 'B t D -> B 1 t', 'mean')  # [B, t, D] => [B, 1, t]
                # Return tuple of both actions (difference will be part of the loss)
                action_output = (action_output_patches, action_output_tokens)
            else:
                raise ValueError(f'Invalid action_output type: {self.params["action_output"]["type"]}')

        else:
            # GAP is default behavior, if not specified
            in_memory = rearrange(sequence_act, 'B t D -> B D t')  # [B, t, D] => [B, D, t]
            action_output = torch.mean(in_memory, dim=1, keepdim=True)  # [B, D, t] -> [B, 1, len(TARGETS)]

        return action_output

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
        encoded_obs = self.pos_embedding(encoded_obs) if not g_conf.LEARNABLE_POS_EMBED else encoded_obs + self.tfx_encoder.pos_embedding # [B, S*cam*H*W/P^2 + K, D]]

        # Add learnable sensor embedding
        if g_conf.SENSOR_EMBED:
            encoded_obs = encoded_obs + self.sensor_embedding  # [B, S*cam*H*W/P^2 + K, D]

        # Pass on to the Transformer encoder
        # Do Stochastic Depth if specified
        if self.params['encoder_embedding']['perception']['vit'].get('stochastic_depth', False):
            in_memory = torchvision.ops.StochasticDepth(p=0.1, mode='row')(self.tfx_encoder.forward(encoded_obs))
        else:
            in_memory = self.tfx_encoder.forward(encoded_obs, mask=self.tfx_mask)  # [B, S*cam*H*W/P^2 + K, D]

        # Get the action output
        action_output = self.action_prediction(in_memory, cam)  # [B, t=2]

        if g_conf.CMD_SPD_TOKENS and g_conf.PREDICT_CMD_SPD:
            command_prediction = self.command_output(in_memory[:, 0])  # [B, 1]
            speed_prediction = self.speed_output(in_memory[:, 1])  # [B, 1]
            return action_output, (command_prediction, speed_prediction)
        return action_output

    def forward_eval(self,
                     s: List[List[torch.Tensor]],
                     s_d: List[torch.Tensor],
                     s_s: List[torch.Tensor],
                     attn_rollout: bool = False,
                     zero_diagonal_attnmap: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)  # Number of frames per camera in sequence
        B = s_d[0].shape[0]  # Batch size; will change if using multiple GPUs
        cam = len(g_conf.DATA_USED)  # Number of cameras

        encoded_obs = self.encode_observations(s, s_d, s_s)  # [B, S*cam*H*W/P^2 + K, D]

        # Add positional encoding
        encoded_obs = self.pos_embedding(encoded_obs) if not g_conf.LEARNABLE_POS_EMBED else encoded_obs + self.tfx_encoder.pos_embedding  # [B, S*cam*H*W/P^2 + K, D]]

        # Add learnable sensor embedding
        if g_conf.SENSOR_EMBED:
            encoded_obs = encoded_obs + self.sensor_embedding  # [B, S*cam*H*W/P^2 + K, D]

        # Pass on to the Transformer encoder
        # [B, S*cam*H*W/P^2 + K, D], num_layers * [B, S*cam*H*W/P^2 + K, S*cam*H*W/P^2 + K]
        in_memory, attn_weights = self.tfx_encoder.forward_return_attn(encoded_obs, mask=self.tfx_mask)

        # Get the action output
        action_output = self.action_prediction(in_memory, cam)  # [B, t=2]

        # Attention stuff
        if attn_rollout:
            attn_weights = utils.attn_rollout(attn_weights)

        if zero_diagonal_attnmap:
            attn_weights = utils.zero_diagonal_att_map(attn_weights)

        # Get the attention weights in the right shape
        if not g_conf.NO_ACT_TOKENS:
            if g_conf.CMD_SPD_TOKENS:
                # We'd like to analyze the attention weights of the [CMD] and [SPD] tokens
                attn_weights_t2t = torch.zeros(B, self.act_tokens_pos[-1]+1, self.act_tokens_pos[-1]+2, device=attn_weights[-1].device)  # [B, t+2, t+3]
                attn_weights_t2t[:, :self.act_tokens_pos[-1]+1, :self.act_tokens_pos[-1]+1] = attn_weights[-1][:, :self.act_tokens_pos[-1]+1, :self.act_tokens_pos[-1]+1]
                attn_weights_t2t[:, :, -1:] = attn_weights[-1][:, :self.act_tokens_pos[-1]+1, -self.tfx_num_patches*S*cam:].sum(dim=-1).unsqueeze(-1)  # [B, t+2, t+3]
                attn_weights_stracc = attn_weights[-1][:, self.act_tokens_pos, -self.tfx_num_patches*S*cam:]  # [B, t, S*cam*(H//P)^2]
                # Normalize the attention weights to be in the range [0, 1], row-wise
                attn_weights_t2t = attn_weights_t2t / attn_weights_t2t.sum(dim=2, keepdim=True)  # [B, t+2, t+3]
                attn_weights_stracc = (attn_weights_stracc - attn_weights_stracc.min()) / (attn_weights_stracc.max() - attn_weights_stracc.min())  # [B, t, S*cam*(H//P)^2]
                attn_weights_stracc = rearrange(attn_weights_stracc, 'B T (S cam h w) -> B T h (S cam w)',
                                                S=S, cam=cam, h=self.tfx_num_patches_h)
                # Give as a tuple
                attn_weights = (attn_weights_t2t, attn_weights_stracc)  # [B, t+2, t+3], [B, t+2, H//P, S*cam*W//P]
            else:
                # Return only the attention weights of the last layer for the [ACC] and [STR] tokens
                attn_weights = attn_weights[-1][:, self.act_tokens_pos, -self.tfx_num_patches*S*cam:]  # [B, t, S*cam*(H//P)^2]
                # Normalize the attention weights to be in the range [0, 1]
                attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())  # [B, t+2, S*cam*(H//P)^2]
                # Rearrange the attention weights to be in the right shape
                attn_weights = rearrange(attn_weights, 'B T (S cam h w) -> B T h (S cam w)', S=S, cam=cam, h=self.tfx_num_patches_h)  # [B, t, H//P, S*cam*W//P]

        else:
            # We don't have any extra tokens, so let's just return the average attention weights of the last layer
            attn_weights = attn_weights[-1].mean(dim=1)  # [B, S*cam*(H//P)^2] or [B, N]
            attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())  # [B, S*cam*(H//P)^2] or [B, N]
            attn_weights = rearrange(attn_weights, 'B (S cam h w) -> B 1 h (S cam w)', S=S, cam=cam, h=self.tfx_num_patches_h)  # [B, 1, H//P, S*cam*W//P]

        return action_output, attn_weights