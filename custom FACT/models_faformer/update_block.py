import torch
import torch.nn as nn
from .base_block import Block
from . import loss
from .loss import MatchCriterion

class UpdateBlock(Block):
    def __init__(self, cfg, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bu

        # Pass projected dim explicitly
        self.frame_branch = self.create_fbranch(cfg, in_dim=cfg.hid_dim)

        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)

        # abranch
        self.action_branch = self.create_abranch(cfg)

        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # a->f
        action_feature = self.f2a_layer(frame_feature, action_feature, X_pos=frame_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass + 1)

        # f->a
        frame_feature = self.a2f_layer(action_feature, frame_feature, X_pos=action_pos, Y_pos=frame_pos)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation
        self.frame_clogit = frame_clogit
        self.action_clogit = action_clogit
        self.f2a_attn = self.f2a_layer.attn[0]
        self.a2f_attn = self.a2f_layer.attn[0]
        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)
        return frame_feature, action_feature

    def compute_loss(self, criterion: MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss(match, torch.transpose(self.f2a_attn_logit, 1, 2), dim=1)
        a2f_loss = criterion.cross_attn_loss(match, self.a2f_attn_logit, dim=2)

        # temporal smoothing loss
        al = loss.smooth_loss(self.a2f_attn_logit)
        fl = loss.smooth_loss(torch.transpose(self.f2a_attn_logit, 1, 2))
        frame_clogit = torch.transpose(self.frame_clogit, 0, 1)  # f, 1, c -> 1, f, c
        l = loss.smooth_loss(frame_clogit)
        smooth_loss = al + fl + l

        return atk_loss + f2a_loss + a2f_loss + frame_loss + self.cfg.Loss.sw * smooth_loss
