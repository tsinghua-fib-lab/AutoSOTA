from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from peft import LoraConfig

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from projects.samtok.models import QWEN3VL_VQSAM2Model, DirectResize
from projects.samtok.datasets import Qwen3VLDataset, qwen25vl_vqsam2_collate_fn

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
qwen3vl_path = "Qwen/Qwen3-VL-4B-MT256x2"
pretrained_pth = None

work_dir = "work_dirs/qwen3vl_4b_toy"

# Scheduler & Optimizer
batch_size = 4  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

model_max_length=8192


#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################

model = dict(
    type=QWEN3VL_VQSAM2Model,
    qwen3vl_hf_model=dict(
        type=Qwen3VLForConditionalGeneration.from_pretrained,
        pretrained_model_name_or_path=qwen3vl_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    tokenizer=dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=qwen3vl_path,
        cache_dir="./cache",
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    ),
    preprocessor=dict(
        type=AutoProcessor.from_pretrained,
        pretrained_model_name_or_path=qwen3vl_path,
    ),
    llm_lora=dict(
        type=LoraConfig,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    ),
    pretrained_pth=pretrained_pth,
    unfreeze_vision_encoder=False,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

sam2_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

standard_dataset = dict(
    type=Qwen3VLDataset,
    tokenizer=dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=qwen3vl_path,
        cache_dir="./cache",
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    ),
    data_args=dict(
        model_type="qwen3vl",
        dataset_use="mask_generation_gres%400",
        max_pixels=2048*28*28,
        min_pixels=4*28*28,
        video_max_total_pixels=576*28*28,
        video_min_total_pixels=144*28*28,
        video_max_frames=8,
        video_min_frames=4,
        video_min_pixels=256 * 28 * 28,
        video_max_pixels=1024 * 28 * 28,
        video_fps=2,
        image_processor=dict(
            type=AutoProcessor.from_pretrained,
            pretrained_model_name_or_path=qwen3vl_path,
        ),
    ),
    sam_preprocessor=sam2_image_processor,
)


train_dataset = dict(
    type=ConcatDataset, datasets=[
        standard_dataset,
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
    collate_fn=dict(type=qwen25vl_vqsam2_collate_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)


# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
# visualizer = None
from mmengine.visualization import Visualizer, TensorboardVisBackend
visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

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