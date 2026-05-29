import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import copy
import math
from .layers import SlidingWindowTransformerEncoder
from .utils import add_positional_encoding, _get_clones, _get_activation_fn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, empty=False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.empty = empty
        self.__compute_pe__(d_model, max_len)

    def __compute_pe__(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)

        if not self.empty:
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)
    
    def __str__(self):
        if self.empty:
            return f"PositionalEncoding(EMPTY)"
        else:
            return f"PositionalEncoding(Dim={self.d_model}, MaxLen={self.max_len})"

    def __repr__(self):
        return str(self)

    def forward(self, x):
        if x.size(0) > self.pe.shape[0]: 
            self.__compute_pe__(self.d_model, x.size(0)+10)
            self.pe = self.pe.to(x.device)

        return self.pe[:x.size(0), :]

class SALayer(nn.Module):
    def __init__(self, q_dim, nhead, dim_feedforward=2048, kv_dim=None,
                 dropout=0.1, attn_dropout=0.1,
                 activation="relu", vpos=False):
        super().__init__()

        kv_dim = q_dim if kv_dim is None else kv_dim
        self.multihead_attn = nn.MultiheadAttention(q_dim, nhead, kdim=kv_dim, vdim=kv_dim, dropout=attn_dropout)

        self.linear1 = nn.Linear(q_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, q_dim)

        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.use_vpos = vpos
        self.dropout_rate = (dropout, attn_dropout)

    def __str__(self) -> str:
        return f"SALayer( q({self.q_dim})xkv({self.kv_dim})->{self.q_dim}, head:{self.nhead}, ffdim:{self.dim_feedforward}, dropout:{self.dropout_rate}, vpos:{self.use_vpos} )"
    
    def __repr__(self):
        return str(self)

    def forward(self, tgt, key, value, query_pos=None, key_pos=None, value_pos=None):
        query = add_positional_encoding(tgt, query_pos)
        key = add_positional_encoding(key, key_pos)
        if self.use_vpos:
            value = add_positional_encoding(value, value_pos)

        tgt2, self.attn = self.multihead_attn(query, key, value, average_attn_weights=False)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt

class SADecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, decoder_layer: SALayer, num_layers, norm=None, in_map=False):
        super().__init__()
        self.in_map = in_map
        if in_map:
            self.in_linear = nn.Linear(in_dim, hid_dim)
        else:
            assert in_dim == hid_dim
        self.layers = _get_clones(decoder_layer, num_layers)
        self.out_linear = nn.Linear(hid_dim, out_dim)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, pos: Optional[Tensor] = None):
        if self.in_map:
            output = self.in_linear(tgt)
        else:
            output = tgt

        for layer in self.layers:
            output = layer(output, output, output, query_pos=pos, key_pos=pos, value_pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        output = self.out_linear(output)

        return output

class TransformerFrameBranch(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        t_cfg = cfg.T
        self.encoder = SlidingWindowTransformerEncoder(
            num_layers=t_cfg.num_layers,
            r1=t_cfg.r1,
            r2=t_cfg.r2,
            num_f_maps=t_cfg.num_f_maps,
            input_dim=in_dim,
            channel_masking_rate=t_cfg.channel_masking_rate,
            alpha=t_cfg.alpha
        )
        self.conv_out = nn.Conv1d(t_cfg.num_f_maps, out_dim, 1)
        
        self.string = f"TransformerFrameBranch(h:{in_dim}->{t_cfg.num_f_maps}x{t_cfg.num_layers}->{out_dim})"

    def __str__(self):
        return self.string

    def __repr__(self):
        return str(self)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        mask = torch.ones_like(x[:, :1, :])
        feature = self.encoder(x, mask)
        out = self.conv_out(feature)
        out = out.permute(2, 0, 1)
        return out
