import torch
import torch.nn as nn
from einops import rearrange, reduce, einsum
import importlib

from configs import g_conf
from _utils import utils
from network.models.building_blocks import FC
from network.models.building_blocks.PositionalEncoding import PositionalEncoding
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoder
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoderLayer


class CIL_multiview(nn.Module):
    def __init__(self, params, rank: int = 0):
        super(CIL_multiview, self).__init__()
        self.params = params
        self.act_tokens_pos = None

        resnet_module = importlib.import_module('network.models.building_blocks.resnet_FM')
        resnet_module = getattr(resnet_module, params['encoder_embedding']['perception']['res']['name'])
        self.encoder_embedding_perception = resnet_module(pretrained=g_conf.IMAGENET_PRE_TRAINED,
                                                          layer_id=params['encoder_embedding']['perception']['res']['layer_id'])
        _, self.res_out_dim, self.res_out_h, self.res_out_w = self.encoder_embedding_perception.get_backbone_output_shape([g_conf.BATCH_SIZE] + g_conf.IMAGE_SHAPE)[params['encoder_embedding']['perception']['res'][ 'layer_id']]

        # Get the sequence length
        self.sequence_length = len(g_conf.DATA_USED) * g_conf.ENCODER_INPUT_FRAMES_NUM * self.res_out_h * self.res_out_w

        if not g_conf.NO_ACT_TOKENS:
            # Add the STR and ACC tokens as parameters
            self.tfx_steer_token = nn.Parameter(torch.empty(1, 1, self.params['TxEncoder']['d_model']).normal_(std=0.02))
            self.tfx_accel_token = nn.Parameter(torch.empty(1, 1, self.params['TxEncoder']['d_model']).normal_(std=0.02))
            self.sequence_length += 2
            self.act_tokens_pos = [0, 1]

        if g_conf.CMD_SPD_TOKENS:
            self.sequence_length += 2
            if self.act_tokens_pos is not None:
                self.act_tokens_pos = [2, 3]


        if self.params['TxEncoder']['learnable_pe']:
            self.positional_encoding = nn.Parameter(torch.zeros(1, self.sequence_length, self.params['TxEncoder']['d_model']))
        else:
            self.positional_encoding = PositionalEncoding(d_model=self.params['TxEncoder']['d_model'], dropout=0.0,
                                                          max_len=self.sequence_length)

        # Sensor embedding is useful when adding different sensors to the sequence
        if g_conf.SENSOR_EMBED:
            self.sensor_embedding = nn.Parameter(torch.empty(1, self.sequence_length, self.tfx_hidden_dim).normal_(std=0.02))  # from BERT

        join_dim = self.params['TxEncoder']['d_model']
        self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, self.params['TxEncoder']['d_model'])
        self.speed = nn.Linear(1, self.params['TxEncoder']['d_model'])

        tx_encoder_layer = TransformerEncoderLayer(d_model=self.params['TxEncoder']['d_model'],
                                                   nhead=self.params['TxEncoder']['n_head'],
                                                   norm_first=self.params['TxEncoder']['norm_first'], batch_first=True)
        self.tx_encoder = TransformerEncoder(tx_encoder_layer, num_layers=self.params['TxEncoder']['num_layers'],
                                             norm=nn.LayerNorm(self.params['TxEncoder']['d_model']))

        self.action_output = FC(params={'neurons': [join_dim] + self.params['action_output']['fc']['neurons'] + [len(g_conf.TARGETS)],
                                        'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                                        'end_layer': True})

        if 'type' in self.params['action_output']:
            if self.params['action_output']['type'] == 'decoder_mlp':
                from network.models.building_blocks.Transformer.TransformerDecoder import TransformerDecoderLayer
                from network.models.building_blocks.Transformer.TransformerDecoder import TransformerDecoder

                self.tfx_decoder_layer = TransformerDecoderLayer(
                    d_model=self.params['action_output']['TxDecoder']['d_model'],
                    nhead=self.params['action_output']['TxDecoder']['n_head'],
                    norm_first=self.params['action_output']['TxDecoder']['norm_first'],
                    batch_first=True)

                self.tfx_decoder = TransformerDecoder(self.tfx_decoder_layer,
                                                      num_layers=self.params['action_output']['TxDecoder']['num_layers'],
                                                      norm=nn.LayerNorm(self.params['action_output']['TxDecoder']['d_model']))

                self.action_query = nn.Parameter(torch.empty(1, 1, self.params['action_output']['TxDecoder']['d_model']).normal_(std=0.02))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        self.train()

    def encode_observations(self, s, s_d, s_s):
        """ Encode the observations into the representation space """
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d[0].shape[0]
        cam = len(g_conf.DATA_USED)  # Number of cameras

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1) # [B, S, cam, 3, H, W]
        x = rearrange(x, 'B S cam C H W -> (B S cam) C H W')
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # Image encoding into tokens for the Encoder input sequence
        e_p, resnet_inter = self.encoder_embedding_perception(x)  # [B*S*cam, dim, h, w]
        e_p = rearrange(e_p, '(B S cam) dim h w -> B (S cam h w) dim', S=S, cam=cam)  # [B, S*cam*h*w, D]

        # Add extra tokens, if sppecified
        if not g_conf.NO_ACT_TOKENS:
            n = e_p.shape[0]  # B
            first_tokens = [self.tfx_steer_token.expand(n, -1, -1), self.tfx_accel_token.expand(n, -1, -1)]
        else:
            first_tokens = []

        # Concatenate the first tokens to the image embeddings
        e_p = torch.cat([*first_tokens, e_p], dim=1)  # [B, S*cam*h*w + K, D], K = 2 if tokens are used

        # Embed the commands and speed
        e_d = self.command(d).unsqueeze(1)  # [B, 1, D]
        e_s = self.speed(s).unsqueeze(1)  # [B, 1, D]

        # Add the embeddings to the image embeddings
        if g_conf.CMD_SPD_TOKENS:
            encoded_obs = torch.cat([e_d, e_s, e_p], dim=1)  # [B, S*cam*H*W/P^2 + K, D]; K = 4 w/tokens else 2
        else:
            encoded_obs = e_p + e_d + e_s  # [B, S*cam*H*W/P^2 + K), D]; K = 2 w/tokens else 0

        return encoded_obs, resnet_inter

    def action_prediction(self, sequence: torch.Tensor, cam: int) -> torch.Tensor:
        """ Predict the action from the encoded sequence """
        # Only use the action tokens [ACT] for the prediction
        if 'action_output' in self.params and self.params['action_output'].get('type', None) is not None:
            action_output_type = self.params['action_output']['type']
            if action_output_type in ['baseline1_patches2act', 'gapn_mlp']:
                # Get the patch representation at the final layer
                patches = reduce(sequence[:, -cam * self.res_out_h * self.res_out_w:], 'B N D -> B D', 'mean')  # [B, N, D] => [B, D]
                # Pass the patch representation through an MLP
                action_output = self.action_output(patches).unsqueeze(1)  # [B, D] => [B, 1, t]
            elif action_output_type == 'decoder_mlp':
                # We pass the whole sequence to the Decoder, and then the output query to an MLP
                # Output shapes: [B, 1, D], num_layers * [B, nhead, 1, 1], num_layers * [B, nhead, 1, N]
                out, sa_weights, mha_weights = self.tfx_decoder(self.action_query.repeat(sequence.shape[0], 1, 1), sequence)
                action_output = self.action_output(out.squeeze()).unsqueeze(1)  # [B, D] => [B, 1, t]
            elif action_output_type == 'baseline2_gapd':
                # Get the action tokens and perform average pooling over the dimension D
                action_output = reduce(sequence[:, self.act_tokens_pos], 'B t D -> B 1 t', 'mean')  # t = 2
            else:
                raise ValueError(f'Invalid action_output type: {self.params["action_output"]["type"]}')

        else:
            # gapn_mlp by default: get the patch representation at the final layer
            patches = reduce(sequence[:, -cam * self.res_out_h * self.res_out_w:], 'B N D -> B D', 'mean')  # [B, N, D] => [B, D]
            # Pass the patch representation through an MLP
            action_output = self.action_output(patches).unsqueeze(1)  # [B, D] => [B, 1, t]

        return action_output

    def forward(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d[0].shape[0]
        cam = len(g_conf.DATA_USED)  # Number of cameras

        encoded_obs, _ = self.encode_observations(s, s_d, s_s)  # [B, S*cam*h*w + K, D]

        # Add positional embedding (fixed or learnable)
        if self.params['TxEncoder']['learnable_pe']:
            pe = encoded_obs + self.positional_encoding    # [B, S*cam*h*w, 512]
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer encoder multi-head self-attention layers
        in_memory, _ = self.tx_encoder(pe)  # [B, S*cam*h*w, 512]

        # Get the action output
        action_output = self.action_prediction(in_memory, cam)  # [B, 1, t=len(TARGETS)]

        return action_output         # (B, 1, 1), (B, 1, len(TARGETS))

    def forward_eval(self, s, s_d, s_s, attn_rollout: bool = False, attn_refinement: bool = False):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d[0].shape[0]
        cam = len(g_conf.DATA_USED)  # Number of cameras

        encoded_obs, resnet_inter = self.encode_observations(s, s_d, s_s)  # [B, S*cam*h*w + K, D]

        if self.params['TxEncoder']['learnable_pe']:
            # positional encoding
            pe = encoded_obs + self.positional_encoding    # [B, S*cam*h*w, 512]
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer encoder multi-head self-attention layers
        in_memory, attn_weights = self.tx_encoder(pe)  # [B, S*cam*h*w, 512]

        # Get the action output
        action_output = self.action_prediction(in_memory, cam)  # [B, 1, t=len(TARGETS)]

        # Attention stuff
        if attn_rollout:
            attn_weights = utils.attn_rollout(attn_weights)

        # Get the attention weights in the right shape
        if not g_conf.NO_ACT_TOKENS:
            if g_conf.CMD_SPD_TOKENS:
                # We'd like to analyze the attention weights of the [CMD] and [SPD] tokens
                attn_weights_t2t = attn_weights[-1][:, :self.act_tokens_pos[-1] + 1, :self.act_tokens_pos[-1] + 1]  # [B, t+2, t+2]
                attn_weights_stracc = attn_weights[-1][:, self.act_tokens_pos, -self.res_out_h * self.res_out_w * S * cam:]  # [B, t, S*cam*(H//P)^2]
                # Normalize the attention weights to be in the range [0, 1], row-wise
                attn_weights_t2t = attn_weights_t2t / attn_weights_t2t.sum(dim=2, keepdim=True)  # [B, t+2, t+2]
                attn_weights_stracc = (attn_weights_stracc - attn_weights_stracc.min()) / (attn_weights_stracc.max() - attn_weights_stracc.min())  # [B, t, S*cam*(H//P)^2]
                attn_weights_stracc = rearrange(attn_weights_stracc, 'B T (S cam h w) -> B T h (S cam w)',
                                                S=S, cam=cam, h=self.res_out_h)
                # Give as a tuple
                attn_weights = (attn_weights_t2t, attn_weights_stracc)  # [B, t+2, t+2], [B, t+2, H//P, S*cam*W//P]
            else:
                # Just work with the last layer
                attn_weights = attn_weights[-1]  # [B, S*cam*H*W/P^2 + K, S*cam*H*W/P^2 + K]
                # Return only the attention weights of the last layer for the [ACC] and [STR] tokens
                attn_act = attn_weights[:, self.act_tokens_pos, -self.res_out_h * self.res_out_w * S * cam:]  # [B, t, S*cam*(H//P)^2]
                # Normalize the attention weights to be in the range [0, 1]
                attn_act = (attn_act - attn_act.min(dim=2, keepdim=True).values) / (attn_act.max(dim=2, keepdim=True).values - attn_act.min(dim=2, keepdim=True).values)  # [B, t+2, S*cam*(H//P)^2]
                if attn_refinement:
                    attn_p2p = attn_weights[:, -self.res_out_h * self.res_out_w * S * cam:, -self.res_out_h * self.res_out_w * S * cam:]  # [B, S*cam*(H//P)^2, S*cam*(H//P)^2]
                    attn_p2p = rearrange(attn_p2p, 'b (n1 n2) (n3 n4) -> b n1 n2 n3 n4', n1=self.res_out_h, n3=self.res_out_h)  # [B, H//P, S*cam*W//P, H//P, S*cam*W//P]
                    attn_act = rearrange(attn_act, 'b t (n1 n2) -> b t n1 n2', n1=self.res_out_h)  # [B, t, H//P, S*cam*W//P]
                    attn_act = einsum(attn_p2p, attn_act, 'b h w h1 w1, b t h1 w1 -> b t h w')
                    attn_act = rearrange(attn_act, 'b t h w -> b t (h w)')  # [B, t, S*cam*(H//P)^2]

                # Rearrange the attention weights to be in the right shape
                attn_weights = rearrange(attn_act, 'B T (S cam h w) -> B T h (S cam w)', S=S, cam=cam, h=self.res_out_h)  # [B, t, H//P, S*cam*W//P]

        else:
            # We don't have any extra tokens, so let's just return the average attention weights of the last layer
            attn_weights = attn_weights[-1].mean(dim=1)  # [B, S*cam*(H//P)^2] or [B, N]
            attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())  # [B, S*cam*(H//P)^2] or [B, N]
            attn_weights = rearrange(attn_weights, 'B (S cam h w) -> B 1 h (S cam w)', S=S, cam=cam,
                                     h=self.res_out_h)  # [B, 1, H//P, S*cam*W//P]

        return action_output, resnet_inter, attn_weights

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
