import math
from yacs.config import CfgNode as CN


_C = CN()
# random seed number
_C.SEED = 0
# number of gpus per node
_C.NUM_GPUS = 8
_C.VISIBLE_DEVICES = 0
# directory to save result txt file
_C.RESULT_DIR = 'results/'
_C.NORMALIZE = 'NST'

_C.DATA_LOADER = CN()
_C.DATA_LOADER.NUM_WORKERS = 2
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.DROP_LAST = True

_C.DATA = CN()
_C.DATA.BASE_DIR = 'data/'
_C.DATA.NAME = 'weather'
_C.DATA.N_VAR = 21
_C.DATA.SEQ_LEN = 96
_C.DATA.LABEL_LEN = 48
_C.DATA.PRED_LEN = 96
_C.DATA.FEATURES = 'M'
_C.DATA.TIMEENC = 0
_C.DATA.FREQ = 'h'
_C.DATA.SCALE = "standard"  # standard, min-max
_C.DATA.TRAIN_RATIO = 0.7
_C.DATA.TEST_RATIO = 0.2
_C.DATA.DATE_IDX = 0
_C.DATA.TARGET_START_IDX = 0
_C.DATA.PERIOD_LEN = 24  # Used only when SAN is ENABLED
_C.DATA.STATION_TYPE = 'adaptive'  # Used only when SAN is ENABLED

_C.TRAIN = CN()
_C.TRAIN.ENABLE = False
_C.TRAIN.SPLIT = 'train'
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.SHUFFLE = True
_C.TRAIN.DROP_LAST = True
# directory to save checkpoints
_C.TRAIN.CHECKPOINT_DIR = 'results/'
# path to checkpoint to resume training
_C.TRAIN.RESUME = ''
# epoch period to evaluate on a validation set
_C.TRAIN.EVAL_PERIOD = 5
# iteration frequency to print progress meter
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.BEST_METRIC_INITIAL = float("inf")
_C.TRAIN.BEST_LOWER = True

_C.VAL = CN()
_C.VAL.SPLIT = 'val'
_C.VAL.BATCH_SIZE = 256
_C.VAL.SHUFFLE = False
_C.VAL.DROP_LAST = False
_C.VAL.VIS = False

_C.TEST = CN()
_C.TEST.ENABLE = True
_C.TEST.SPLIT = 'test'
_C.TEST.BATCH_SIZE = 256
_C.TEST.SHUFFLE = False
_C.TEST.DROP_LAST = False

_C.TTA = CN()
_C.TTA.ENABLE = False
_C.TTA.MODULE_NAMES_TO_ADAPT = 'cali'  # all, norm, etc
_C.TTA.LOG = False
_C.TTA.SOLVER = CN()
_C.TTA.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.TTA.SOLVER.BASE_LR = 0.005
_C.TTA.SOLVER.WEIGHT_DECAY = 0.0001
_C.TTA.SOLVER.MOMENTUM = 0.9
_C.TTA.SOLVER.NESTEROV = True
_C.TTA.SOLVER.DAMPENING = 0.0
_C.TTA.TAFAS = CN()
_C.TTA.TAFAS.PAAS = True
_C.TTA.TAFAS.PERIOD_N = 1
_C.TTA.TAFAS.BATCH_SIZE = 64
_C.TTA.TAFAS.STEPS = 1
_C.TTA.TAFAS.ADJUST_PRED = True
_C.TTA.TAFAS.CALI_MODULE = True
_C.TTA.TAFAS.GATING_INIT = 0.01
_C.TTA.TAFAS.HIDDEN_DIM = 128
_C.TTA.TAFAS.GCM_VAR_WISE = True

_C.MODEL = CN()
_C.MODEL.NAME = 'iTransformer'
_C.MODEL.task_name = 'long_term_forecast'
_C.MODEL.seq_len = _C.DATA.SEQ_LEN 
_C.MODEL.label_len = _C.DATA.LABEL_LEN # Not needed in iTransformer
_C.MODEL.pred_len = _C.DATA.PRED_LEN 
_C.MODEL.e_layers = 4
_C.MODEL.d_layers = 1 # Not needed in iTransformer
_C.MODEL.factor = 3 # Not used in iTransformer Full Attention. Used in Prob Attention (probabilistic attention) in informer
_C.MODEL.enc_in = _C.DATA.N_VAR # Used only in classification
_C.MODEL.dec_in = _C.DATA.N_VAR # Not needed in iTransformer
_C.MODEL.c_out = _C.DATA.N_VAR # Not needed in iTransformer
_C.MODEL.d_model = 512 # embedding dimension
_C.MODEL.d_ff = 512  # feedforward dimension d_model -> d_ff -> d_model
_C.MODEL.moving_avg = 25
_C.MODEL.output_attention = False # whether the attention weights are returned by the forward method of the attention class
_C.MODEL.dropout = 0.1
_C.MODEL.n_heads = 8
_C.MODEL.activation = 'gelu'
_C.MODEL.channel_independence = True
_C.MODEL.METRIC_NAMES = ('MAE',)
_C.MODEL.LOSS_NAMES = ('MSE',)
_C.MODEL.embed = 'timeF'
_C.MODEL.freq = 'h'
_C.MODEL.ignore_stamp = False
# OLS params
_C.MODEL.instance_norm = True
_C.MODEL.individual = False
_C.MODEL.alpha = 0.000001

_C.NORM_MODULE = CN()
_C.NORM_MODULE.ENABLE = False  # NST
_C.NORM_MODULE.NAME = 'SAN'  # SAN, RevIN, DishTS

_C.SAN = CN()
_C.SAN.RESULT_DIR = 'results/station/'
_C.SAN.TRAIN = CN()
_C.SAN.TRAIN.CHECKPOINT_DIR = 'results/station/'
_C.SAN.SOLVER = CN()
_C.SAN.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.SAN.SOLVER.START_EPOCH = 0
_C.SAN.SOLVER.MAX_EPOCH = 10
_C.SAN.SOLVER.BASE_LR = 0.001
_C.SAN.SOLVER.WEIGHT_DECAY = 0.0001
_C.SAN.SOLVER.MOMENTUM = 0.9
_C.SAN.SOLVER.NESTEROV = True
_C.SAN.SOLVER.DAMPENING = 0.0
_C.SAN.SOLVER.LR_POLICY = 'cosine'
_C.SAN.SOLVER.COSINE_END_LR = 0.0
_C.SAN.SOLVER.COSINE_AFTER_WARMUP = False
_C.SAN.SOLVER.WARMUP_EPOCHS = 0
_C.SAN.SOLVER.WARMUP_START_LR = 0.001

_C.REVIN = CN()
_C.REVIN.EPS = 1e-5
_C.REVIN.AFFINE = True
_C.REVIN.RESULT_DIR = 'results/revin/'
_C.REVIN.TRAIN = CN()
_C.REVIN.TRAIN.CHECKPOINT_DIR = 'results/revin/'

_C.DISHTS = CN()
_C.DISHTS.INIT = 'standard'  # standard, avg, uniform
_C.DISHTS.RESULT_DIR = 'results/dishts/'
_C.DISHTS.TRAIN = CN()
_C.DISHTS.TRAIN.CHECKPOINT_DIR = 'results/dishts/'

_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCH = 50
_C.SOLVER.OPTIMIZING_METHOD = 'adam'
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = True
_C.SOLVER.DAMPENING = 0.0
_C.SOLVER.LR_POLICY = 'cosine'
_C.SOLVER.COSINE_END_LR = 0.0
_C.SOLVER.COSINE_AFTER_WARMUP = False
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.WARMUP_START_LR = 0.001

_C.WANDB = CN()
_C.WANDB.ENABLE = False
_C.WANDB.PROJECT = 'TAFAS'
_C.WANDB.NAME = ''
_C.WANDB.JOB_TYPE = ''
_C.WANDB.NOTES = ''
_C.WANDB.DIR = './'
_C.WANDB.SET_LOG_DIR = True


def get_cfg_defaults():

    return _C.clone()


def get_norm_module_cfg(cfg):
    return getattr(cfg, cfg.NORM_MODULE.NAME.upper())


def get_norm_method(cfg):
    assert cfg.NORM_MODULE.NAME in ('RevIN', 'SAN', 'DishTS')
    norm_method = cfg.NORM_MODULE.NAME if cfg.NORM_MODULE.ENABLE else 'NST'
    return norm_method
