import torch
import torch.nn as nn
from . import basic

class Block(nn.Module):
    """
    Base Block class for common functions
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        lines = f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.action_branch},\n  a2f:{self.a2f_layer if hasattr(self, 'a2f_layer') else None},\n  f2a:{self.f2a_layer if hasattr(self, 'f2a_layer') else None}\n)"
        return lines

    def __repr__(self):
        return str(self)

    def process_feature(self, feature, nclass):
        # use the last several dimension as logit of action classes
        clogit = feature[:, :, -nclass:]  # class logit
        feature = feature[:, :, :-nclass]  # feature without clogit
        cprob = basic.logit2prob(clogit, dim=-1)  # apply softmax
        feature = torch.cat([feature, cprob], dim=-1)

        return feature, clogit

    def create_fbranch(self, cfg, in_dim=None, f_inmap=False):
        if in_dim is None:
            in_dim = cfg.f_dim

        if cfg.f == 'm':  # use MSTCN
            frame_branch = basic.MSTCN(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers,
                                      dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        elif cfg.f == 'm2':  # use MSTCN++
            frame_branch = basic.MSTCN2(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers,
                                       dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        elif cfg.f == 't':  # ### NEW ### use Transformer
            out_dim = cfg.hid_dim + self.nclass
            frame_branch = basic.TransformerFrameBranch(cfg, in_dim, out_dim)
        else:
            raise ValueError(f"Unknown frame branch type: {cfg.f}")

        return frame_branch

    def create_abranch(self, cfg):
        if cfg.a == 'sa':  # self-attention layers, for update blocks
            l = basic.SALayer(cfg.a_dim, cfg.a_nhead, dim_feedforward=cfg.a_ffdim, dropout=cfg.dropout, attn_dropout=cfg.dropout)
            action_branch = basic.SADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, l, cfg.a_layers, in_map=False)
        elif cfg.a == 'sca':  # self+cross-attention layers, for input blocks when video transcripts are not available
            layer = basic.SCALayer(cfg.a_dim, cfg.hid_dim, cfg.a_nhead, cfg.a_ffdim, dropout=cfg.dropout)
            norm = torch.nn.LayerNorm(cfg.a_dim)
            action_branch = basic.SCADecoder(cfg.a_dim, cfg.a_dim, cfg.hid_dim, layer, cfg.a_layers, in_map=False)
        elif cfg.a in ['gru', 'gru_om']:  # GRU, for input blocks when video transcripts are available
            assert self.cfg.FACT.trans
            out_map = (cfg.a == 'gru_om')
            action_branch = basic.ActionUpdate_GRU(cfg.a_dim, cfg.a_dim, cfg.hid_dim, cfg.a_layers, dropout=cfg.dropout, out_map=out_map)
        else:
            raise ValueError(cfg.a)

        return action_branch

    def create_cross_attention(self, cfg, outdim, kq_pos=True):
        # one layer of cross-attention for cross-branch communication
        layer = basic.X2Y_map(cfg.hid_dim, cfg.hid_dim, outdim,
                              head_dim=cfg.hid_dim,
                              dropout=cfg.dropout, kq_pos=kq_pos)
        return layer

    @staticmethod
    def _eval(action_clogit, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)

        action_clogit = action_clogit.squeeze(1)
        a2f_attn = a2f_attn.squeeze(0)
        qtk_cpred = action_clogit.argmax(1)
        null_cid = action_clogit.shape[-1] - 1
        action_loc = torch.where(qtk_cpred != null_cid)[0]

        if len(action_loc) == 0:
            return fbranch_prob.argmax(1)

        qtk_prob = torch.softmax(action_clogit[:, :-1], dim=1)  # remove logit of null classes
        action_pred = a2f_attn[:, action_loc].argmax(-1)
        action_pred = action_loc[action_pred]
        abranch_prob = qtk_prob[action_pred]

        prob = (1 - weight) * abranch_prob + weight * fbranch_prob
        return prob.argmax(1)

    @staticmethod
    def _eval_w_transcript(transcript, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)
        fbranch_prob = fbranch_prob[:, transcript]

        N = len(transcript)
        a2f_attn = a2f_attn[0, :, :N]  # 1, f, a -> f, s'
        abranch_prob = torch.softmax(a2f_attn, dim=-1)  # f, s'

        prob = (1 - weight) * abranch_prob + weight * fbranch_prob
        pred = prob.argmax(1)  # f
        pred = transcript[pred]
        return pred

    def eval(self, transcript=None):
        if not self.cfg.FACT.trans:
            return self._eval(self.action_clogit, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)
        else:
            return self._eval_w_transcript(transcript, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)
