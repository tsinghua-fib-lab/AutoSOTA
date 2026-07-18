import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Public default paths are repo-relative. Local validation can override these
# paths through environment variables without committing local symlinks.
DATA_DIR = os.environ.get(
    "RECYCLING4VLALIGNMENT_DATA_DIR", os.path.join(REPO_ROOT, "data")
)
WEIGHTS_DIR = os.environ.get(
    "RECYCLING4VLALIGNMENT_WEIGHTS_DIR", os.path.join(REPO_ROOT, "weights")
)
EMBEDDINGS_DIR = os.environ.get(
    "RECYCLING4VLALIGNMENT_EMBEDDINGS_DIR", os.path.join(DATA_DIR, "embeddings")
)
CHECKPOINT_DIR = os.environ.get(
    "RECYCLING4VLALIGNMENT_CHECKPOINT_DIR",
    os.path.join(REPO_ROOT, "aligner_checkpoints"),
)

# Seed
GLOBAL_SEED = 42

# Training weights and their descriptions
PARAMS_DIR = WEIGHTS_DIR

# Training params
OPTIMIZER_NAME = 'SGD'
SGD_DECAY_FACTOR = 0.1
SGD_DECAY_THRESHOLD = 0.1
SGD_PATIENCE = 7
SGD_WARMUP_STEPS = 1000
SGD_WARMUP_UPDATE_INTERVAL = 200
SGD_INITIAL_LR = 0.001
ADAM_PATIENCE = 10
ADAM_INITIAL_LR = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 0.001
ADAM_EPS = 1e-8
ADAM_DECAY_FACTOR = 0.5
ADAM_DECAY_THRESHOLD = 0.1
LBFGS_MAX_ITER = 20000
LBFGS_MAX_EVAL = None  # If None, defaults to max_iter * 1.25
LBFGS_TOLERANCE_GRAD = 1e-07
LBFGS_TOLERANCE_CHANGE = 1e-09
LBFGS_HISTORY_SIZE = 10
BALANCED_LOSS = True

# Regularization params
L2_NORM_REG_WEIGHT = 0.001  # Weight for L2 norm regularization penalty

# Performance optimizations
USE_MIXED_PRECISION = True  # Enable mixed precision training
NUM_WORKERS = 2  # Set to 0 for maximum parallel capacity with embeddings
PIN_MEMORY = True  # Pin memory for faster GPU transfer
PERSISTENT_WORKERS = False  # Disable for embeddings to reduce overhead

# Model Lists
# ===========

# Torchvision model names (ImageNet-1K pretrained models)
TORCHVISION_MODEL_NAMES = {
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # 'resnext50_32x4d', 'resnext101_32x8d', 
    'densenet121', 'densenet169', 'densenet201',
    # 'mobilenet_v2',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7'
    # 'regnet_x_200mf', 'regnet_x_400mf', 'regnet_x_600mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',
    # 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
}

# HuggingFace model names (ImageNet-21K/22K pretrained but with ImageNet-1K classification heads)
# Note: These models have 1000 classes for ImageNet-1K classification, not 21K classes
HUGGINGFACE_MODEL_NAMES = [
    # Google ViT models (ImageNet-21K pretrained with ImageNet-1K classification heads)
    'google/vit-base-patch16-224',
    'google/vit-large-patch16-224',
    # Microsoft Swin models (ImageNet-22K pretrained with ImageNet-1K classification heads)
    'microsoft/swin-tiny-patch4-window7-224',
    'microsoft/swin-base-patch4-window7-224',
    'timm/beit_base_patch16_224.in22k_ft_in22k',
    'timm/caformer_s18.sail_in22k',
    'timm/convformer_s18.sail_in22k',
    'timm/convnext_base.fb_in22k',
    'timm/eva02_base_patch14_448.mim_in22k_ft_in22k',
    'timm/swin_base_patch4_window12_384.ms_in22k',
    'timm/tiny_vit_21m_224.dist_in22k'
]


IMAGENET1K_HEAD_MODELS = [
    'timm/beit_base_patch16_224.in22k_ft_in22k_in1k',
    'timm/caformer_s18.sail_in22k_ft_in1k',
    'timm/convformer_s18.sail_in22k_ft_in1k',
    'timm/convnext_base.fb_in22k_ft_in1k',
    'timm/eva02_base_patch14_448.mim_in22k_ft_in1k',
    'timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k',
    'timm/tiny_vit_21m_224.dist_in22k_ft_in1k',
    'timm/resnetv2_50x1_bit.goog_in21k_ft_in1k',
    'timm/resnetv2_101x1_bit.goog_in21k_ft_in1k',
    'timm/vit_base_patch32_224.augreg_in21k_ft_in1k',
    'timm/vit_base_patch16_224.augreg_in21k_ft_in1k'
]

IMAGENET21K_HEAD_MODELS = [
    'timm/beit_base_patch16_224.in22k_ft_in22k',
    'timm/caformer_s18.sail_in22k',
    'timm/convformer_s18.sail_in22k',
    'timm/convnext_base.fb_in22k',
    'timm/eva02_base_patch14_448.mim_in22k_ft_in22k',
    'timm/swin_base_patch4_window12_384.ms_in22k',
    'timm/tiny_vit_21m_224.dist_in22k'
]

IMAGENET21K_2EXTRA_HEAD_MODELS = [
    'timm/resnetv2_50x1_bit.goog_in21k',
    'timm/resnetv2_101x1_bit.goog_in21k',
    'timm/vit_base_patch32_224_in21k',
    'timm/vit_base_patch16_224_in21k'
]

INATURALIST_HEAD_MODELS = [
    'timm/convnext_large_mlp.laion2b_ft_augreg_inat21',
    'timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21'
]

# Text model names for multimodal alignment
TEXT_MODEL_NAMES = [
    'clip_vitb32',
    'all-roberta-large-v1', 
    'all-mpnet-base-v2',
    'all-MiniLM-L6-v2',
    "sup-simcse-bert-base-uncased",
    "unsup-simcse-bert-base-uncased"
]


IMAGENET21K_WNIDS_PATH = os.path.join(DATA_DIR, 'imagenet21k_wnids.txt')
IMAGENET21K_2EXTRA_WNIDS_PATH = os.path.join(DATA_DIR, 'imagenet21k_2extra_wnids.txt')
IMAGENET1K_WNIDS_PATH = os.path.join(DATA_DIR, 'imagenet1k', 'imagenet1k_wnids.txt')


SUPPORTED_DATASETS = ["resisc45", "eurosat", "flowers102", "oxfordpets", "food101", "cifar10", "cifar100", "dtd", "places365"]

FLICKR30_CAPTIONS_PATH = os.path.join(DATA_DIR, 'flickr30k', 'captions.txt')
TEST_FLICKR30_CAPTIONS_PATH = os.path.join(DATA_DIR, 'flickr30k', 'flickr30k_test_karpathy.txt')


VALID_TEXT_MODELS = [
    'clip_vitl14_336px',
    'clip_vitb32',
    'all-roberta-large-v1',
    'all-mpnet-base-v2',
    'all-roberta-large-coco-contrastive',
    'all-MiniLM-L6-v2',
    "sup-simcse-bert-base-uncased",
    "unsup-simcse-bert-base-uncased"
]

VALID_IMAGE_MODELS = [
    'clip_vitl14_336px',
    'clip_vitb32',
    'timm/beit_base_patch16_224.in22k_ft_in22k_in1k',
    'timm/caformer_s18.sail_in22k_ft_in1k',
    'timm/convformer_s18.sail_in22k_ft_in1k',
    'timm/convnext_base.fb_in22k_ft_in1k',
    'timm/eva02_base_patch14_448.mim_in22k_ft_in1k',
    'timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k',
    'timm/tiny_vit_21m_224.dist_in22k_ft_in1k',
    'timm/resnetv2_50x1_bit.goog_in21k_ft_in1k',
    'timm/resnetv2_101x1_bit.goog_in21k_ft_in1k',
    'timm/vit_base_patch32_224.augreg_in21k_ft_in1k',
    'timm/vit_base_patch16_224.augreg_in21k_ft_in1k',
    'timm/resnetv2_50x1_bit.goog_in21k',
    'timm/resnetv2_101x1_bit.goog_in21k',
    'timm/vit_base_patch32_224_in21k',
    'timm/vit_base_patch16_224_in21k',
    'timm/beit_base_patch16_224.in22k_ft_in22k',
    'timm/caformer_s18.sail_in22k',
    'timm/convformer_s18.sail_in22k',
    'timm/convnext_base.fb_in22k',
    'timm/swin_base_patch4_window12_384.ms_in22k',
    'timm/tiny_vit_21m_224.dist_in22k',
    'timm/eva02_base_patch14_448.mim_in22k_ft_in22k',
    'timm/convnext_large_mlp.laion2b_ft_augreg_inat21',
    'timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21'
]

VALID_CLASSIFICATION_DATASETS = [
    "resisc45", "eurosat", "mnist", "flowers102", "oxfordpets", 
    "food101", "cifar10", "cifar100", "dtd", "places365", "ham10000"
    ]
VALID_RETRIEVAL_DATASETS = ["flickr30k", "coco"]
