from fvcore.common.config import CfgNode
from loguru import logger

import clip
from dataset.transforms import AUGMENTATIONS

_C = CfgNode()

# ---------------------------------------------------------------------------- #
# general options
_C.RNG_SEED = 0
_C.EVAL_INTERVAL = 1
_C.LOG_INTERVAL = 10
_C.SAVE_INTERVAL = None
_C.TRAIN_PRECISION = "fp16"
_C.SAVE_PATH = None
_C.RESUME_CHECKPOINT = None
_C.LOG_TO_WANDB = False
_C.EVAL_ONLY = False
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Model options
_C.MODEL = CfgNode()
_C.MODEL.VIZ_BACKBONE = "ViT-B/16"
_C.MODEL.PER_CLS_PROMPTS = False  # CoOp option
_C.MODEL.PROMPT_POSITION = "start"  # CoOp option, position of the class names
_C.MODEL.IMG_CONDITIONING = False  # CoCoop option
_C.MODEL.NUM_PROMPTS = 16  # CoOp option
_C.MODEL.FRAME_AGGREGATION = (
    "transformer_2"  # # Video option, mean, max, transformer_nlayers
)
_C.MODEL.SOFTMAX_TEMP = None  # if None, use CLIP's temp
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Data options
_C.DATA = CfgNode()

_C.DATA.TYPE = "image"  # image or video
_C.DATA.DATASET_NAME = "ImageNet"
_C.DATA.DATA_PATH = ""
_C.DATA.MEAN = [0.48145466, 0.4578275, 0.40821073]
_C.DATA.STD = [0.26862954, 0.26130258, 0.27577711]

_C.DATA.TRAIN_AUGS = ["random_resized_crop", "random_flip", "normalize"]
_C.DATA.TRAIN_RESIZE = None
_C.DATA.TRAIN_CROP_SIZE = 224
_C.DATA.TEST_AUGS = ["resize", "center_crop", "normalize"]
_C.DATA.TEST_RESIZE = 224
_C.DATA.TEST_CROP_SIZE = 224
_C.DATA.TEST_STRIDES = [8]

# Video data options
_C.DATA.NUM_FRAMES = 8
_C.DATA.TARGET_FPS = 30
_C.DATA.TRAIN_STRIDES = [8]
_C.DATA.TRAIN_VIDEO_SAMPLER = "random"
_C.DATA.TEST_NUM_CLIPS = 1
# single_view, multi_view_strides, multi_view_sliding
_C.DATA.TEST_METHOD = "single_view"

# Fewshot / Zeroshot options
_C.DATA.FEWSHOT = False  # Video option, for images just set N_SHOT > 0
_C.DATA.ZEROSHOT = False  # Video option
_C.DATA.USE_ALL_CLASSES = False  # Video option
_C.DATA.N_SHOT = 0  # Video option
_C.DATA.C_WAY = -1  # Image and video option
_C.DATA.N_QUERY_SHOT = 95  # Video option
_C.DATA.USE_BASE_AND_NEW = False  # Image option
# Image option, domain generalization
# ImageNet-A R V2 and Sketch
_C.DATA.TARGET_DATASET = None
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# data loader options
_C.DATALOADER = CfgNode()

_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.TRAIN_BATCHSIZE = 8
_C.DATALOADER.TEST_BATCHSIZE = 8
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Optimizer options
_C.OPT = CfgNode()
_C.OPT.MAX_EPOCHS = 1
_C.OPT.LR = 0.1
_C.OPT.TYPE = "sgd"
_C.OPT.LR_POLICY = "cosine"
_C.OPT.COSINE_END_LR = 0.0
_C.OPT.LINEAR_END_LR = 0.0

_C.OPT.STEPS = []
_C.OPT.WEIGHT_DECAY = 1e-4
_C.OPT.WARMUP_EPOCHS = 0.0
_C.OPT.ZERO_WD_1D_PARAM = False
_C.OPT.CLIP_L2_GRADNORM = None
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #


def check_and_update_configs(cfg):
    assert cfg.TRAIN_PRECISION in ["fp32", "fp16", "amp"]

    assert cfg.MODEL.VIZ_BACKBONE in clip.available_models()
    assert cfg.MODEL.PROMPT_POSITION in ["start", "middle", "end"]
    assert (
        cfg.MODEL.NUM_PROMPTS < 65
    ), "Possibly no room left for class name and other tokens"
    assert cfg.MODEL.FRAME_AGGREGATION.split(
        "_")[0] in ["mean", "max", "transformer"]

    assert cfg.DATA.TRAIN_VIDEO_SAMPLER in ["random", "center"]

    assert cfg.OPT.TYPE in ["sgd", "adam", "adamw"]
    assert cfg.OPT.LR_POLICY in ["cosine", "step", "linear", "constant"]
    if cfg.OPT.CLIP_L2_GRADNORM:
        assert cfg.OPT.CLIP_L2_GRADNORM > 0.0

    assert cfg.DATA.TEST_METHOD in [
        "single_view",
        "multi_view_strides",
        "multi_view_sliding",
    ]
    if cfg.DATA.TEST_METHOD == "single_view":
        assert cfg.DATA.TEST_NUM_CLIPS == 1 and len(cfg.DATA.TEST_STRIDES) == 1
        cfg.DATA.TEST_VIDEO_SAMPLER = "center"
    elif cfg.DATA.TEST_METHOD == "multi_view_strides":
        assert cfg.DATA.TEST_NUM_CLIPS == len(cfg.DATA.TEST_STRIDES)
        cfg.DATA.TEST_VIDEO_SAMPLER = "center"
    else:
        assert len(cfg.DATA.TEST_STRIDES) == 1 and cfg.DATA.TEST_NUM_CLIPS > 1
        cfg.DATA.TEST_VIDEO_SAMPLER = "sliding"

    for aug in cfg.DATA.TRAIN_AUGS:
        assert aug in list(AUGMENTATIONS.keys()
                           ), f"Augmentation {aug} is not supported"

    if cfg.DATA.FEWSHOT:
        assert cfg.DATA.TYPE == "video"
        assert cfg.DATA.DATASET_NAME in [
            "UCF101",
            "HMDB51",
            "K400",
        ], "Few-shot dataset is only supported for UCF101, HMDB51, K400"
        cfg.DATA.DATASET_NAME = f"{cfg.DATA.DATASET_NAME}FewShot"

    if cfg.DATA.ZEROSHOT:
        assert cfg.DATA.TYPE == "video"
        assert cfg.DATA.DATASET_NAME in [
            "UCF101",
            "HMDB51",
            "K700",
        ], "Zero-shot dataset is only supported for UCF101, HMDB51, K700"
        cfg.DATA.DATASET_NAME = f"{cfg.DATA.DATASET_NAME}ZeroShot"

    if cfg.DATA.USE_ALL_CLASSES:
        assert cfg.DATA.TYPE == "video"
        assert cfg.DATA.C_WAY == -1, "C_WAY must be -1 if USE_ALL_CLASSES is True"
        assert (
            cfg.DATA.FEWSHOT
        ), "C way (using all classes) is only supported for fewshot"

    assert (
        cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
    ), "For clip, use the same as the original size for both"

    if cfg.RESUME_CHECKPOINT:
        assert cfg.RESUME_CHECKPOINT.endswith(".pyth")

    return cfg


def get_cfg(args):
    cfg = _C.clone()

    # update default with ones from yaml file
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    # update using the passed values to argparse
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg = check_and_update_configs(cfg)

    logger.info(f"Configs of this run:\n{cfg}")

    return cfg
