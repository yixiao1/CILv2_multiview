'''
import copy
import torch.nn as nn
import torch.nn.functional as F
#from network.models.building_blocks.Transformer.MultiheadAttention import MultiheadAttention

class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask= None, memory_mask= None, tgt_key_padding_mask= None,
                memory_key_padding_mask= None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        attn_layers_sa = []
        attn_layers_mha = []
        for mod in self.layers:
            output, attn_output_weights_sa,  attn_output_weights_mha= mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attn_layers_sa.append(attn_output_weights_sa)
            attn_layers_mha.append(attn_output_weights_mha)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_layers_sa, attn_layers_mha


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation=F.relu,
                 layer_norm_eps=1e-5, norm_first=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask= None,
                tgt_key_padding_mask= None, memory_key_padding_mask= None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            sa_block, attn_output_weights_sa = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + sa_block
            mha_block, attn_output_weights_mha = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + mha_block
            x = x + self._ff_block(self.norm3(x))
        else:
            sa_block, attn_output_weights_sa = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + sa_block)
            mha_block, attn_output_weights_mha = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + mha_block)
            x = self.norm3(x + self._ff_block(x))

        return x, attn_output_weights_sa, attn_output_weights_mha


    # self-attention block
    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x, attn_output_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        return self.dropout1(x), attn_output_weights

    # multihead attention block
    def _mha_block(self, x, mem,
                   attn_mask, key_padding_mask):
        x, attn_output_weights = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x), attn_output_weights

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from network.models.building_blocks.blocks import SinusoidalPosEmb


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
                tgt, 
                memory, 
                tgt_mask= None, 
                memory_mask= None, 
                tgt_key_padding_mask= None,
                memory_key_padding_mask= None,
                pos=None,
                query_pos=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            pos: positional enc. for MEMORY (added to K in cross-attn).
            query_pos: positional enc. for QUERIES (added to Q/K in self-attn, to Q in cross-attn).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        attn_layers_sa = []
        attn_layers_mha = []
        for mod in self.layers:
            output, attn_output_weights_sa,  attn_output_weights_mha = mod(
                output, 
                memory, 
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos)
            attn_layers_sa.append(attn_output_weights_sa)
            attn_layers_mha.append(attn_output_weights_mha)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_layers_sa, attn_layers_mha


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, 
                tgt, 
                memory, 
                tgt_mask = None, 
                memory_mask= None,
                tgt_key_padding_mask= None, 
                memory_key_padding_mask= None,
                pos=None,
                query_pos=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            pos: positional enc. for memory (added to K in cross-attn)
            query_pos: positional enc. for queries (added to Q/K in self-attn; to Q in cross-attn)

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            sa_block, attn_output_weights_sa = self._sa_block(self.norm1(x), query_pos, tgt_mask, tgt_key_padding_mask)
            x = x + sa_block
            mha_block, attn_output_weights_mha = self._mha_block(self.norm2(x), memory, query_pos, pos, memory_mask, memory_key_padding_mask)
            x = x + mha_block
            x = x + self._ff_block(self.norm3(x))
        else:
            sa_block, attn_output_weights_sa = self._sa_block(x, query_pos, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + sa_block)
            mha_block, attn_output_weights_mha = self._mha_block(x, memory, query_pos, pos, memory_mask, memory_key_padding_mask)
            x = self.norm2(x + mha_block)
            x = self.norm3(x + self._ff_block(x))

        return x, attn_output_weights_sa, attn_output_weights_mha


    # self-attention block (q = k = x + query_pos, v = x)
    def _sa_block(self, 
                  x,
                  query_pos,
                  attn_mask, 
                  key_padding_mask):
        q = x if query_pos is None else x + query_pos
        k = x if query_pos is None else x + query_pos
        v = x
        out, attn_output_weights = self.self_attn(
            q, k, v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return self.dropout1(out), attn_output_weights

    # cross-attention block (q = x + query_pos, k = mem + pos, v = mem)
    def _mha_block(self, 
                   x, 
                   mem,
                   query_pos,
                   pos,
                   attn_mask, 
                   key_padding_mask):
        q = x if query_pos is None else x + query_pos
        k = mem if pos is None else mem + pos
        v = mem
        out, attn_output_weights = self.multihead_attn(
            q, k, v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        return self.dropout2(out), attn_output_weights

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



class TransformerDecoderForDiffusion(nn.Module):
    """
    DP-based (https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py) diffusion transformer that uses CIL++ TransformerDecoder

    Expects:
      - `tx_decoder`: an instantiated CIL++ TransformerDecoder.
      - `cond` at forward time: (B, To, D) tokens already produced by CIL++ encoder
        (we use To=1: the current-image token compressed by the core).
      - `sample`: noisy actions (B, T, input_dim).
      - `timestep`: scalar or (B,).

    Internals:
      - Embeds action tokens -> (B, T, D) and adds learnable pos_emb(T).
      - Builds memory = [ time_token || cond ] and adds learnable cond_pos_emb(2).
      - Runs **CIL++ TransformerDecoder** with tgt=(embedded actions) and memory.
      - LayerNorm + Linear head -> (B, T, output_dim).
    """
    def __init__(self,
                 tx_decoder: TransformerDecoder,
                 d_model: int,
                 input_dim: int,
                 output_dim: int,
                 horizon: int,
                 n_obs_steps: int = 1,          # To=1 (current only)
                 p_drop_emb: float = 0.1
        ):
        super().__init__()

        self.tx_decoder = tx_decoder
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps  # we use 1

        # DP stems
        self.input_emb = nn.Linear(self.input_dim, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.horizon, self.d_model))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = SinusoidalPosEmb(self.d_model)  # (B,) -> (B,D)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + self.n_obs_steps, self.d_model))  # [time || obs]
        
        # Output head
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.output_dim)

        # Init (DP-like)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, 
                sample: torch.tensor, # (B, T, input_dim), noisy actions
                timestep: Union[torch.Tensor, float, int], # (B,) or scalar
                cond: torch.tensor  # (B, To=1, D), obs token from CIL++ encoder
                ) -> torch.tensor:
        """
        sample: (B, T, input_dim)  noisy action tokens
        timestep: (B,) or int      diffusion step
        cond: (B, To, cond_dim)    conditioning tokens (time token is added internally)
        return: (B, T, output_dim) per-token prediction (e.g., noise epsilon)
        """
        # time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B,1,n_emb)

        # Memory = [time || obs]
        memory = torch.cat([time_emb, cond], dim=1)     # (B, 1+To=2, D)
        memory = self.drop(memory + self.cond_pos_emb[:, :memory.shape[1], :])

        # Tgt = embedded noisy actions + pos
        tgt = self.drop(self.input_emb(sample) + self.pos_emb[:, : self.horizon, :])  # (B,T,D)

        # Decode with CIL++ TransformerDecoder
        dec_out, _, _ = self.tx_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None
        )

        out = self.head(self.ln_f(dec_out))  # (B,T,output_dim)
        return out
    