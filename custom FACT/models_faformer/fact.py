"""
FACT - Feature-Action Coupling in Time module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import basic
from utils import utils
from configs.utils import update_from
from . import loss
from .loss import MatchCriterion
from .base_block import Block
from .input_block import InputBlock
from .update_block import UpdateBlock
from .update_block_tdu import UpdateBlockTDU
from .basic import torch_class_label_to_segment_label, time_mask

class FACT(nn.Module):
    def __init__(self, cfg, in_dim, n_classes):
        super().__init__()
        self.cfg = cfg
        self.num_classes = n_classes

        # Project input frame features from in_dim (e.g., 523) to hid_dim (e.g., 512)
        self.input_proj = nn.Linear(in_dim, cfg.hid_dim)

        base_cfg = cfg.Bi
        self.frame_pe = basic.PositionalEncoding(base_cfg.hid_dim, max_len=10000, empty=(not cfg.FACT.fpos))
        self.channel_masking_dropout = nn.Dropout2d(p=cfg.FACT.cmr)

        if not cfg.FACT.trans:
            self.action_query = nn.Parameter(torch.randn([cfg.FACT.ntoken, 1, base_cfg.a_dim]))
        else:
            self.action_pe = basic.PositionalEncoding(base_cfg.a_dim, max_len=1000)
            self.action_embed = nn.Embedding(n_classes, base_cfg.a_dim)

        # block configuration
        block_list = []
        for i, t in enumerate(cfg.FACT.block):
            if t == 'i':
                # Pass projected dimension cfg.hid_dim here
                block = InputBlock(cfg, cfg.hid_dim, n_classes)
            elif t == 'u':
                update_from(cfg.Bu, base_cfg, inplace=True)
                base_cfg = cfg.Bu
                # Pass projected dimension cfg.hid_dim here
                block = UpdateBlock(cfg, n_classes)
            elif t == 'U':
                update_from(cfg.BU, base_cfg, inplace=True)
                base_cfg = cfg.BU
                # Pass projected dimension cfg.hid_dim here
                block = UpdateBlockTDU(cfg, n_classes)
            else:
                raise ValueError(f"Unknown block type: {t}")

            block_list.append(block)

        self.block_list = nn.ModuleList(block_list)
        self.mcriterion = None

    def _forward_one_video(self, seq, transcript=None):
        # seq shape: (T, B=1, in_dim)
        # Project input features to hid_dim
        seq = self.input_proj(seq)  # Now (T, B=1, hid_dim)

        frame_feature = seq
        frame_pe = self.frame_pe(seq)
        if self.cfg.FACT.cmr:
            frame_feature = frame_feature.permute([1, 2, 0])
            frame_feature = self.channel_masking_dropout(frame_feature)
            frame_feature = frame_feature.permute([2, 0, 1])

        if self.cfg.TM.use and self.training:
            frame_feature = time_mask(frame_feature,
                                      self.cfg.TM.t, self.cfg.TM.m, self.cfg.TM.p,
                                      replace_with_zero=True)

        if not self.cfg.FACT.trans:
            action_pe = self.action_query  # M, B=1, H
            action_feature = torch.zeros_like(action_pe)
        else:
            action_pe = self.action_pe(transcript)
            action_feature = self.action_embed(transcript).unsqueeze(1)
            action_feature = action_feature + action_pe
            action_pe = torch.zeros_like(action_pe)

        block_output = []
        for block in self.block_list:
            frame_feature, action_feature = block(frame_feature, action_feature, frame_pe, action_pe)
            block_output.append([frame_feature, action_feature])

        return block_output

    def _loss_one_video(self, label):
        mcriterion: MatchCriterion = self.mcriterion
        mcriterion.set_label(label)

        block : Block = self.block_list[-1]
        cprob = basic.logit2prob(block.action_clogit, dim=-1)
        match = mcriterion.match(cprob, block.a2f_attn)

        ######## per block loss
        loss_list = []
        for block in self.block_list:
            loss = block.compute_loss(mcriterion, match)
            loss_list.append(loss)

        self.loss_list = loss_list
        final_loss = sum(loss_list) / len(loss_list)
        return final_loss

    def forward(self, seq_list, label_list, compute_loss=False):
        save_list = []
        final_loss = []

        for i, (seq, label) in enumerate(zip(seq_list, label_list)):
            seq = seq.unsqueeze(1)
            trans = torch_class_label_to_segment_label(label)[0]
            self._forward_one_video(seq, trans)

            pred = self.block_list[-1].eval(trans)
            save_data = {'pred': utils.to_numpy(pred)}
            save_list.append(save_data)

            if compute_loss:
                loss = self._loss_one_video(label)
                final_loss.append(loss)
                save_data['loss'] = {'loss': loss.item()}

        if compute_loss:
            final_loss = sum(final_loss) / len(final_loss)
            return final_loss, save_list
        else:
            return save_list

    def save_model(self, fname):
        torch.save(self.state_dict(), fname)
