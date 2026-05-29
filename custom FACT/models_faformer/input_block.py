import torch
import torch.nn as nn
from .base_block import Block
from . import loss
from .loss import MatchCriterion

class InputBlock(Block):
    def __init__(self, cfg, in_dim, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.Bi

        self.frame_branch = self.create_fbranch(cfg, in_dim, f_inmap=True)
        self.action_branch = self.create_abranch(cfg)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos, action_clogit=None):
        # frame branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # action branch
        action_feature = self.action_branch(action_feature, frame_feature, pos=frame_pos, query_pos=action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass + 1)

        # save features for loss and evaluation
        self.frame_clogit = frame_clogit
        self.action_clogit = action_clogit

        return frame_feature, action_feature

    def compute_loss(self, criterion: MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        atk_loss = criterion.action_token_loss(match, self.action_clogit)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1)
        smooth_loss = loss.smooth_loss(frame_clogit)

        return frame_loss + atk_loss + self.cfg.Loss.sw * smooth_loss
