import torch
import torch.nn as nn
from .base_block import Block
from .loss import MatchCriterion
from . import loss
from . import basic
from utils import utils

class UpdateBlockTDU(Block):
    """
    Update Block with Temporal Downsampling and Upsampling
    """

    def __init__(self, cfg, nclass):
        super().__init__()
        self.cfg = cfg
        self.nclass = nclass

        cfg = cfg.BU

        # Pass projected dim explicitly
        self.frame_branch = self.create_fbranch(cfg, in_dim=cfg.hid_dim)

        # layers for temporal downsample and upsample
        self.seg_update = nn.GRU(cfg.hid_dim, cfg.hid_dim // 2, cfg.s_layers, bidirectional=True)
        self.seg_combine = nn.Linear(cfg.hid_dim, cfg.hid_dim)

        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(cfg, cfg.a_dim)

        # abranch
        self.action_branch = self.create_abranch(cfg)

        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(cfg, cfg.f_dim)

        # layers for temporal downsample and upsample
        self.sf_merge = nn.Sequential(nn.Linear((cfg.hid_dim + cfg.f_dim), cfg.f_dim), nn.ReLU())

    def temporal_downsample(self, frame_feature):
        # get action segments based on predictions
        cprob = frame_feature[:, :, -self.nclass:]
        _, pred = cprob[:, 0].max(dim=-1)
        pred = utils.to_numpy(pred)
        segs = utils.parse_label(pred)

        tdu = basic.TemporalDownsampleUpsample(segs)
        tdu.to(cprob.device)

        # downsample frames to segments
        seg_feature = tdu.feature_frame2seg(frame_feature)

        # refine segment features
        seg_feature, hidden = self.seg_update(seg_feature)
        seg_feature = torch.relu(seg_feature)
        seg_feature = self.seg_combine(seg_feature)  # combine forward and backward features
        seg_feature, seg_clogit = self.process_feature(seg_feature, self.nclass)

        return tdu, seg_feature, seg_clogit

    def temporal_upsample(self, tdu, seg_feature, frame_feature):
        # upsample segments to frames
        s2f = tdu.feature_seg2frame(seg_feature)

        # merge with original framewise features to keep low-level details
        frame_feature = self.sf_merge(torch.cat([s2f, frame_feature], dim=-1))

        return frame_feature

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # downsample frame features to segment features
        tdu, seg_feature, seg_clogit = self.temporal_downsample(frame_feature)  # seg_feature: S, 1, H

        # f->a cross attention: segment queries action
        seg_center = torch.LongTensor([int((s.start + s.end) / 2) for s in tdu.segs]).to(seg_feature.device)
        seg_pos = frame_pos[seg_center]
        action_feature = self.f2a_layer(seg_feature, action_feature, X_pos=seg_pos, Y_pos=action_pos)

        # action branch update
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.nclass + 1)

        # a->f cross attention: action queries segment
        seg_feature = self.a2f_layer(action_feature, seg_feature, X_pos=action_pos, Y_pos=seg_pos)

        # upsample segment features back to frame features
        frame_feature = self.temporal_upsample(tdu, seg_feature, frame_feature)

        # frame branch update
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.nclass)

        # save features for loss and evaluation
        self.frame_clogit = frame_clogit
        self.seg_clogit = seg_clogit
        self.tdu = tdu
        self.action_clogit = action_clogit

        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.f2a_attn = tdu.attn_seg2frame(self.f2a_layer.attn[0].transpose(2, 1)).transpose(2, 1)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)
        self.a2f_attn = tdu.attn_seg2frame(self.a2f_layer.attn[0])

        return frame_feature, action_feature

    def compute_loss(self, criterion: MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        seg_loss = criterion.frame_loss_tdu(self.seg_clogit, self.tdu)
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss_tdu(match, torch.transpose(self.f2a_attn_logit, 1, 2), self.tdu, dim=1)
        a2f_loss = criterion.cross_attn_loss_tdu(match, self.a2f_attn_logit, self.tdu, dim=2)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1)
        smooth_loss = loss.smooth_loss(frame_clogit)

        return (frame_loss + seg_loss) / 2 + atk_loss + f2a_loss + a2f_loss + self.cfg.Loss.sw * smooth_loss
