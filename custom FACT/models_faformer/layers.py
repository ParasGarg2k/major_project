import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import copy
from .utils import _get_clones
from .attention import AttLayer

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, nchannels, dropout=0.5, layernorm=True, layernorm_eps=1e-5, ngroup=1):
        super(DilatedResidualLayer, self).__init__()
        self.dilation = dilation
        self.nchannels = nchannels
        self.dropout_rate = dropout

        self.conv_dilated = nn.Conv1d(nchannels, nchannels, 3, padding=dilation, dilation=dilation, groups=ngroup)
        self.conv_1x1 = nn.Conv1d(nchannels, nchannels, 1)
        self.dropout = nn.Dropout(dropout)

        self.use_layernorm = layernorm
        if layernorm:
            self.norm = nn.LayerNorm(nchannels, eps=layernorm_eps)
        else:
            self.norm = None

    def __str__(self):
        return f"DilatedResidualLayer(Conv(d={self.dilation},h={self.nchannels}), 1x1(h={self.nchannels}), Dropout={self.dropout_rate}, ln={self.use_layernorm})"

    def __repr__(self):
        return str(self)

    def forward(self, x, mask=None):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        if mask is not None:
            x = (x + out) * mask[:, 0:1, :]
        else:
            x = x + out

        if self.norm:
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)

        return x

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        att_out = self.att_layer(self.instance_norm(out), f, mask)
        out = self.alpha * att_out + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask

class SlidingWindowTransformerEncoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, channel_masking_rate, alpha):
        super(SlidingWindowTransformerEncoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, 'sliding_att', 'encoder', alpha) for i in range(num_layers)])
        
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        if self.channel_masking_rate > 0 and self.training:
            x = x.unsqueeze(3)
            x = self.dropout(x)
            x = x.squeeze(3)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        return feature
