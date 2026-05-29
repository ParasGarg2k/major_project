from yacs.config import CfgNode as CN

_C = CN()

# auxiliary setting
_C.aux = CN()
_C.aux.gpu = 1
_C.aux.mark = ""
_C.aux.runid = 0
_C.aux.debug = False
_C.aux.wandb_project = "FACT"
_C.aux.wandb_user = ""
_C.aux.wandb_offline = False
_C.aux.resume = "max"
_C.aux.eval_every = 1000
_C.aux.print_every = 200

_C.class_names = [
    "take", "open", "pour", "close", "shake",
    "scoop", "stir", "put", "fold", "spread", "background"
]

# dataset
_C.dataset = "breakfast"
_C.split = "split1"
_C.sr = 1
_C.hid_dim= 512
_C.eval_bg = False
_C.feature_path = None
_C.groundTruth_path = None
_C.map_fname = None
_C.split_path = None
_C.feature_transpose = False
_C.bg_class = 0
_C.average_transcript_len = 0.0

# training
_C.batch_size = 4
_C.optimizer = "SGD"
_C.epoch = 2
_C.lr = 0.1
_C.lr_decay = -1
_C.momentum = 0.009
_C.weight_decay = 0.000
_C.clip_grad_norm = 10.0

# model
_C.FACT = FACT = CN()
FACT.ntoken = 10
FACT.block = "iuUU"
FACT.trans = True
FACT.fpos = True
FACT.cmr = 0.3
FACT.mwt = 0.1
FACT.token_phase = 3
FACT.prompt = True
FACT.dynamic_token = True
FACT.token_thresh = 0.5

_C.BaFormer = BaFormer = CN()
BaFormer.n_queries = 64
BaFormer.hid_dim = 256
BaFormer.lambda_sparse = 0.5
BaFormer.lambda_consistency = 0.1

# Transformer sub-config shared for frame branches
def create_transformer_cfg():
    T = CN()
    T.num_layers = 4
    T.r1 = 1
    T.r2 = 1
    T.num_f_maps = 128
    T.conv_kernel_size = 3
    T.dropout = 0.1
    T.alpha = 1.0
    T.channel_masking_rate= 0.5 
    return T

# input block
_C.Bi = Bi = CN()
Bi.hid_dim = 512
Bi.dropout = 0.5
Bi.a = "sca"
Bi.a_nhead = 8
Bi.a_ffdim = 2048
Bi.a_layers = 6
Bi.a_dim = 512
Bi.f = 'transformer'
Bi.f_layers = 10
Bi.f_ln = True
Bi.f_dim = 512
Bi.f_ngp = 4
Bi.T = create_transformer_cfg()

# update block
_C.Bu = Bu = CN()
Bu.hid_dim = None
Bu.dropout = None
Bu.a = "sa"
Bu.a_nhead = None
Bu.a_ffdim = None
Bu.a_layers = 1
Bu.a_dim = None
Bu.f = 'transformer'
Bu.f_layers = 5
Bu.f_ln = None
Bu.f_dim = None
Bu.f_ngp = None
Bu.T = create_transformer_cfg()

# update block with temporal downsample and upsample
_C.BU = BU = CN()
BU.hid_dim = None
BU.dropout = None
BU.a = "sa"
BU.a_nhead = None
BU.a_ffdim = None
BU.a_layers = 1
BU.a_dim = None
BU.f = 'transformer'
BU.f_layers = 5
BU.f_ln = None
BU.f_dim = None
BU.f_ngp = None
BU.s_layers = 1
BU.T = create_transformer_cfg()

# Loss
_C.Loss = Loss = CN()
Loss.pc = 1.0
Loss.a2fc = 1.0
Loss.match = 'o2o'
Loss.bgw = 1.0
Loss.nullw = -1.0
Loss.sw = 0.0
Loss.temporal_affinity_weight = 0.0

# temporal masking
_C.TM = TM = CN()
TM.use = False
TM.t = 30
TM.p = 0.05
TM.m = 5
TM.inplace = True

_C.upgrades = upgrades = CN()
upgrades.use_temporal_affinity_loss = True
upgrades.attention_supervision = True

def get_cfg_defaults():
    return _C.clone()
