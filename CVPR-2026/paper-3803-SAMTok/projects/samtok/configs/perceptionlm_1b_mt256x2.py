import torch
from torch.optim import AdamW
from mmengine.optim import CosineAnnealingLR, LinearLR, AmpOptimWrapper
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)

from peft import LoraConfig
from transformers import AutoModelForImageTextToText

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.dataset import ConcatDataset

from projects.samtok.models import PerceptionLM_TokenMask
from projects.samtok.datasets import TokenMaskDataset, perceptionlm_collate_fn

#########################################################################
#                             PART 1  Settings                          #
#########################################################################

# Model
mllm_name_or_path = "facebook/Perception-LM-1B-MT256x2"
exp_name = "perceptionlm_1b_mt256x2_toy"
work_dir = f'./work_dirs/{exp_name}'

max_length = 16384
lazy_load = True

# Scheduler & Optimizer
batch_size = 4 # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1 # grad clip
warmup_ratio = 0.03

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################

model = dict(
    type=PerceptionLM_TokenMask,
    mllm=dict(
        type=AutoModelForImageTextToText.from_pretrained,
        pretrained_model_name_or_path=mllm_name_or_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ),
    llm_lora=False,
    visual_encoder_lora=False,
    freeze_llm=False,
    freeze_visual_encoder=True,
    freeze_connector=True,
    unfreeze_vocab=True,
    unfreeze_lm_head=True,
    use_activation_checkpointing=True,
    pretrained_pth=None,
)


#########################################################################
#                    PART 3  Dataset & DataLoader                       #
#########################################################################
tokenmask_dataset = dict(
    type=TokenMaskDataset,
    dataset_use="mask_generation_gres%400",
    model_path=mllm_name_or_path,
    max_num_tiles=8,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        tokenmask_dataset,
    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=perceptionlm_collate_fn)
)

#########################################################################
#                    PART 4  Scheduler & Optimizer                      #
#########################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay,
    ),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype=torch.bfloat16,
)

# learning policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True
    )
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#########################################################################
#                             PART 5  Runtime                           #
#########################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = []

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit
    ),
    # set sampler seed in distributed environment,
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl')
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
