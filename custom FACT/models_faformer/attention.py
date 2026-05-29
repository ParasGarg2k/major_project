import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
from .utils import add_positional_encoding, _get_activation_fn

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        Scalar dot-product attention.
        :param proj_query: (B, C, L)
        :param proj_key: (B, C, L)
        :param proj_val: (B, C, L)
        :param padding_mask: (B, 1, L)
        :return: (B, C, L)
        '''
        B, C, L = proj_query.shape
        _, C_k, _ = proj_key.shape
        assert C == C_k

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = energy / np.sqrt(C)

        padding_mask = padding_mask.permute(0, 2, 1)

        min_len = min(attention.shape[1], padding_mask.shape[1])
        attention = attention[:, :min_len, :min_len]
        padding_mask = padding_mask[:, :min_len, :]

        attention_mask = padding_mask.expand(-1, -1, min_len)
        attention = attention.masked_fill(attention_mask == 0, -1e9)
        attention = self.softmax(attention)

        attention = attention * attention_mask

        proj_val = proj_val[:, :, :min_len]
        out = torch.bmm(proj_val, attention.permute(0, 2, 1))

        return out, attention

class X2Y_map(nn.Module):
    def __init__(self, x_dim, y_dim, y_outdim, head_dim, dropout=0.5, kq_pos=False):
        super(X2Y_map, self).__init__()
        self.kq_pos = kq_pos

        self.X_K = nn.Linear(x_dim, head_dim)
        self.X_V = nn.Linear(x_dim, head_dim)
        self.Y_Q = nn.Linear(y_dim, head_dim)

        self.Y_W = nn.Linear(y_dim+head_dim, y_outdim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X_feature, Y_feature, X_pos=None, Y_pos=None, X_pad_mask=None, Y_pad_mask=None):
        """
        X: x, b, h
        Y: y, b, h
        """
        X = X_feature.shape[0]
        Y = Y_feature.shape[0]

        if (X_pos is not None) and self.kq_pos:
            x = add_positional_encoding(X_feature, X_pos)
            xk = self.X_K(x) 
        else:
            xk = self.X_K(X_feature) 

        xv = self.X_V(X_feature)

        if (Y_pos is not None) and self.kq_pos:
            y = add_positional_encoding(Y_feature, Y_pos)
            yq = self.Y_Q(y)
        else:
            yq = self.Y_Q(Y_feature)

        assert X_pad_mask is None and Y_pad_mask is None

        attn_logit = torch.einsum('xbd,ybd->byx', xk, yq)
        attn_logit = attn_logit / math.sqrt(xk.shape[-1])
        self.attn_logit = attn_logit
        attn = torch.softmax(attn_logit, dim=-1)
        
        attn_feat = torch.einsum('byx,xbh->ybh', attn, xv)
        concat_feature = torch.cat([Y_feature, attn_feat], dim=-1)
        concat_feature = self.dropout(concat_feature)

        Y_feature = self.Y_W(concat_feature)

        self.attn = attn.unsqueeze(1)

        return Y_feature

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type):
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['sliding_att']
        assert self.stage in ['encoder', 'decoder']

        self.att_helper = AttentionHelper()

    def forward(self, x1, x2, mask):
        query = self.query_conv(x1)
        key = self.key_conv(x1)
        value = self.value_conv(x2) if self.stage == 'decoder' else self.value_conv(x1)

        if self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)
        else:
            raise NotImplementedError(f"{self.att_type} not implemented.")

    def _sliding_window_self_att(self, q, k, v, mask):
        B, C_q, L = q.shape
        bl = self.bl
        
        nb = (L + bl - 1) // bl
        padded_len = nb * bl
        pad_len = padded_len - L

        if pad_len > 0:
            q = F.pad(q, (0, pad_len))
            k = F.pad(k, (0, pad_len))
            v = F.pad(v, (0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=0)

        q_windows = q.unfold(dimension=2, size=bl, step=bl)
        k_windows = k.unfold(dimension=2, size=bl, step=bl)
        v_windows = v.unfold(dimension=2, size=bl, step=bl)
        if mask is not None:
            mask_windows = mask.unfold(dimension=2, size=bl, step=bl)
        else:
            mask_windows = None

        q_windows = q_windows.permute(0, 2, 1, 3).reshape(B * nb, C_q, bl)
        k_windows = k_windows.permute(0, 2, 1, 3).reshape(B * nb, C_q, bl)
        v_windows = v_windows.permute(0, 2, 1, 3).reshape(B * nb, C_q, bl)
        if mask_windows is not None:
            mask_windows = mask_windows.reshape(B * nb, 1, bl)

        q_t = q_windows.transpose(1, 2)
        k_t = k_windows.transpose(1, 2)
        v_t = v_windows.transpose(1, 2)

        attn_scores = torch.bmm(q_t, k_t.transpose(1, 2)) / (C_q ** 0.5)

        if mask_windows is not None:
            attn_mask = mask_windows.expand(-1, bl, -1)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_probs, v_t)
        attn_output = attn_output.transpose(1, 2)

        output = self.conv_out(F.relu(attn_output))

        output = output.reshape(B, nb, output.shape[1], bl).permute(0, 2, 1, 3).reshape(B, output.shape[1], nb * bl)

        if pad_len > 0:
            output = output[:, :, :-pad_len]

        return output


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
            x = x.unsqueeze(3) # B, C, T, 1
            x = self.dropout(x)
            x = x.squeeze(3)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        return feature


class TransformerFrameBranch(nn.Module):
    """
    ### NEW ###
    Wrapper for the SlidingWindowTransformerEncoder to make it compatible with FACT's frame branch.
    """
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        # It's recommended to add a 'T' section to your config .yaml file for these parameters
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
        # FACT blocks pass features as (T, B, H)
        # The transformer encoder expects (B, H, T)
        x = x.permute(1, 2, 0)
        
        # Create a mask of all ones since single videos are not padded
        mask = torch.ones_like(x[:, :1, :])
        
        # Get features from the transformer encoder
        feature = self.encoder(x, mask)
        
        # Map features to the output dimension expected by the rest of FACT
        out = self.conv_out(feature)
        
        # Permute back to (T, B, H)
        out = out.permute(2, 0, 1)
        
        return out
