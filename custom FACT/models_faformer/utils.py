import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional

def time_mask(feature, T, num_masks, p, replace_with_zero=False, clone=False):
    """
    T: max drop length - cfg.t
    num_masks: num drop - cfg.m
    p: max drop ratio - cfg.p

    feature: T, B, H
    """
    if clone:
        feature = feature.clone()

    len_spectro = feature.shape[0]
    
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t = min( int(p*len_spectro), t )
        t_zero = random.randrange(0, len_spectro - t)

        if (t_zero == t_zero + t): 
            return feature

        if (replace_with_zero): 
            feature[t_zero:t_zero+t] = 0
        else: 
            feature[t_zero:t_zero+t] = feature.mean()
    return feature

def torch_class_label_to_segment_label(label):
    segment_label = torch.zeros_like(label)
    current = label[0]
    transcript = [label[0]]
    aid = 0
    for i, l in enumerate(label):
        if l == current:
            pass
        else:
            current = l
            aid += 1
            transcript.append(l)
        segment_label[i] = aid

    transcript = torch.LongTensor(transcript).to(label.device)
    
    return transcript, segment_label

def logit2prob(clogit, dim=-1, class_sep=None):
    if class_sep is None or class_sep<=0:
        cprob = torch.softmax(clogit, dim=dim)
    else:
        assert dim==-1, dim
        cprob1 = torch.softmax(clogit[..., :class_sep], dim=dim)
        cprob2 = torch.softmax(clogit[..., class_sep:], dim=dim)
        cprob = torch.cat([cprob1, cprob2], dim=dim)
    
    return cprob

class TemporalDownsampleUpsample():
    def __init__(self, segs):
        self.segs = segs
        self.num_seg = len(segs)

        self.seg_label = []
        for i, seg in enumerate(segs):
            self.seg_label.extend([i]*seg.len)
        self.seg_label = torch.LongTensor(self.seg_label)
        self.seg_lens = torch.LongTensor([s.len for s in segs])

    def cuda(self):
        self.seg_label = self.seg_label.cuda()
        self.seg_lens = self.seg_lens.cuda()

    def to(self, device):
        self.seg_label = self.seg_label.to(device)
        self.seg_lens = self.seg_lens.to(device)

    def feature_frame2seg(self, frame_feature, normalize=True):
        f, b, h = frame_feature.shape
        assert b == 1

        seg_feature = torch.zeros(self.num_seg, b, h, device=frame_feature.device)
        seg_feature.index_add_(0, self.seg_label, frame_feature)

        if normalize:
            seg_feature = seg_feature / self.seg_lens[:, None, None]

        return seg_feature

    def attn_frame2seg(self, frame_attn):
        b, f, a = frame_attn.shape
        assert b == 1

        seg_attn = torch.zeros(b, self.num_seg, a, device=frame_attn.device)
        seg_attn.index_add_(1, self.seg_label, frame_attn)

        seg_attn = seg_attn / self.seg_lens[:, None]

        return seg_attn

    def feature_seg2frame(self, seg_feature):
        """
        seg_feature : S, B, H
        """
        frame_feature = seg_feature[self.seg_label]
        return frame_feature

    def attn_seg2frame(self, seg_attn):
        """
        seg_attn : B, S, A
        """
        assert seg_attn.shape[0] == 1
        frame_attn = seg_attn[0, self.seg_label].unsqueeze(0)
        return frame_attn

def _diff(x, y):
    return (x-y).abs().max()

def add_positional_encoding(tensor, pos):
    return tensor

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
