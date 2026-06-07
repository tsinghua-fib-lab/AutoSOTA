import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import argparse
import copy
import datetime
import faulthandler
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from transformers import PaliGemmaProcessor

sys.path.append(str(Path(__file__).absolute().parents[1]))

from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLMMetrics, VLMMetricsWeb
from prismatic.training.strategies.vlm_fsdp import VLMFSDPStrategy

from hispatial.model import HiSpatialVLM

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

overwatch = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_fsdp_wrap_policy_and_checkpointing(configs):
    if configs["strategy"] == "fsdp_paligemma2":
        from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionTransformer
        from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
        from hispatial.model import CombinedMultiModalProjector, Conv2dForXYZ

        policy = {SiglipEncoderLayer, SiglipVisionTransformer, Gemma2DecoderLayer, CombinedMultiModalProjector, Conv2dForXYZ}
        checkpointing_policy = {Gemma2DecoderLayer} if configs["enable_gradient_checkpointing"] else None
        return policy, checkpointing_policy
    else:
        raise NotImplementedError(f"Unsupported strategy: {configs['strategy']}")


def get_epoch_and_step_from_checkpoint(checkpoint_path):
    if checkpoint_path is None:
        return 0, 0
    try:
        basename = os.path.basename(checkpoint_path)
        arr = basename.split('.')[0].split('-')
        epoch = int(arr[0].split('=')[1])
        step = int(arr[1].split('=')[1])
        return epoch, step
    except Exception as e:
        print(f"Error parsing checkpoint path {checkpoint_path}: {e}")
        return 0, 0


def find_last_checkpoint(checkpoint_dir):
    checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
    if not os.path.exists(os.path.join(checkpoint_dir)):
        return None
    checkpoint_list = os.listdir(os.path.join(checkpoint_dir))
    print(f"All checkpoints:", checkpoint_list)
    last_checkpoint_info = None
    for folder in checkpoint_list:
        if os.path.isdir(os.path.join(checkpoint_dir, folder)):
            files = os.listdir(os.path.join(checkpoint_dir, folder))
            if 'weights.pt' not in files:
                continue
            epoch, step = get_epoch_and_step_from_checkpoint(os.path.join(checkpoint_dir, folder))
            if last_checkpoint_info is None or step > last_checkpoint_info[1]:
                last_checkpoint_info = (os.path.join(checkpoint_dir, folder), step, epoch)
    if last_checkpoint_info is None:
        return None
    return last_checkpoint_info[0]


def load_model_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    print("Checkpoint loaded successfully")
    return model


def posix_to_str(d):
    """Recursively convert Path objects to strings for JSON serialization."""
    if isinstance(d, dict):
        return {k: posix_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [posix_to_str(v) for v in d]
    elif isinstance(d, Path):
        return str(d)
    return d


def deep_update(d1, d2):
    """Recursively merge d2 into d1."""
    for k, v in d2.items():
        if isinstance(v, dict) and isinstance(d1.get(k), dict):
            deep_update(d1[k], d2[k])
        else:
            d1[k] = v
    return d1


def load_config(config_file):
    with open(config_file) as f:
        _config = json.load(f)
    config = {}
    if _config.get("parent"):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def experiment(variant):
    overwatch.info(f"world_size: {overwatch.world_size()}", ctx_level=1)
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # W&B login
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("Please set the WANDB_API_KEY environment variable.")
    wandb.login(key=wandb_api_key)

    os.makedirs(variant["output_root"], exist_ok=True)

    run_id = variant.get("task_name", "train")
    batch_size = variant["batch_size"]
    total_batch_size = variant["total_batch_size"]
    run_id = f"{run_id}_TB{total_batch_size}_B{batch_size}_bf16{variant['use_bf16']}"

    checkpoint_dir = os.path.join(variant["output_root"], run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    copied_variant = posix_to_str(copy.deepcopy(variant))
    if overwatch.rank() == 0:
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(copied_variant, f, indent=2)
        overwatch.info(f"Config saved to {checkpoint_dir}", ctx_level=1)
        print(json.dumps(copied_variant, indent=2))
    dist.barrier()

    # ---- Model ----
    overwatch.info("Loading model", ctx_level=1)
    resume_step = 0
    resume_epoch = 0
    model_load_path = variant["model_load_path"]

    if variant["resume"]:
        if model_load_path is None:
            model_load_path = find_last_checkpoint(checkpoint_dir)
            overwatch.info(f"Found last checkpoint: {model_load_path}", ctx_level=1)
        if model_load_path is not None:
            resume_epoch, resume_step = get_epoch_and_step_from_checkpoint(model_load_path)
            print(f"Resume from {model_load_path}, epoch: {resume_epoch}, step: {resume_step}")

    model = HiSpatialVLM.from_pretrained(
        pretrained_model_name_or_path=variant["backbone_path"],
        attn_implementation="eager",
    )
    model = model.train()
    model.vision_tower.requires_grad_(False)
    model.depth_encoder.requires_grad_(True)
    model.multi_modal_projector.requires_grad_(True)
    model.language_model.requires_grad_(True)

    if variant["resume"] and model_load_path is not None:
        model = load_model_checkpoint(model, os.path.join(model_load_path, "weights.pt"))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    # ---- FSDP Strategy ----
    variant["trainer"]["enable_mixed_precision_training"] = variant["use_bf16"]

    training_strategy = VLMFSDPStrategy(
        vlm=model,
        device_id=overwatch.local_rank(),
        stage=None,
        epochs=variant["trainer"]["max_epochs"],
        max_steps=variant["trainer"]["max_steps"],
        global_batch_size=total_batch_size,
        per_device_batch_size=batch_size,
        learning_rate=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
        max_grad_norm=variant["trainer"]["gradient_clip_val"],
        lr_scheduler_type=variant["lr_scheduler_type"],
        warmup_ratio=variant["warmup_ratio"],
        enable_gradient_checkpointing=variant["trainer"]["enable_gradient_checkpointing"],
        enable_mixed_precision_training=variant["trainer"]["enable_mixed_precision_training"],
        reduce_in_full_precision=variant["trainer"]["reduce_in_full_precision"],
    )

    auto_wrap_policy, checkpointing_policy = get_fsdp_wrap_policy_and_checkpointing(variant["trainer"])

    training_strategy.run_setup_iterable(
        run_dir=checkpoint_dir,
        max_steps=variant["trainer"]["max_steps"],
        auto_wrap_policy_modules=auto_wrap_policy,
        checkpointing_policy_modules=checkpointing_policy,
    )

    if variant["resume"] and model_load_path is not None:
        training_strategy.load_optimizer_and_scheduler(model_load_path)

    # ---- Metrics ----
    trackers = ["wandb"]
    overwatch.info(f"Creating Metrics with Active Trackers => `{trackers}`")
    metrics = VLMMetricsWeb(
        trackers,
        hparams=copied_variant,
        run_id=run_id,
        run_dir=checkpoint_dir,
        wandb_project=variant["wandb_project"],
        wandb_entity=variant["wandb_entity"],
        resume_step=resume_step,
    )

    # ---- Data ----
    overwatch.info("Building dataloader", ctx_level=1)
    num_workers = variant["train_dataset"]["num_workers"]
    prefetch_factor = variant["train_dataset"]["prefetch_factor"]
    overwatch.info(f"num_workers: {num_workers}, prefetch_factor: {prefetch_factor}", ctx_level=1)

    from hispatial.data.vqa_dataset import (
        VQACollateFn,
        GeneralSampleTransform,
        WildSampleTransform,
        CA1MSampleTransform,
        build_vqa_dataloader,
    )
    from hispatial.data.tar_shard_dataset import split_shards_by_node

    vlm_processor = PaliGemmaProcessor.from_pretrained(variant["processor_path"])
    img_size = variant.get("image_size", 448)

    wild_sample_transform = WildSampleTransform(vlm_processor, img_size=img_size)
    ca1m_sample_transform = CA1MSampleTransform(vlm_processor, img_size=img_size)
    general_sample_transform = GeneralSampleTransform(vlm_processor, img_size=img_size)
    collate_fn = VQACollateFn(vlm_processor.tokenizer.pad_token_id, max_length=1500)

    # Gather tar shards
    coyo_dir = Path(variant["train_dataset"]["coyo_dir"])
    o365_dir = Path(variant["train_dataset"]["o365_dir"])
    general_data_dir = Path(variant["train_dataset"]["general_data_dir"])
    ca1m_data_dir = Path(variant["train_dataset"]["ca1m_data_dir"])

    coyo_tar_list = [str(p) for p in coyo_dir.glob("*/*.tar")]
    o365_tar_list = [str(p) for p in o365_dir.glob("*/*.tar")]
    general_tar_list = [str(p) for p in general_data_dir.glob("*/*.tar")]
    ca1m_tar_list = [str(p) for p in ca1m_data_dir.glob("*.tar")]

    print(f"Found {len(coyo_tar_list)} shards in {coyo_dir}")
    print(f"Found {len(o365_tar_list)} shards in {o365_dir}")
    print(f"Found {len(general_tar_list)} shards in {general_data_dir}")
    print(f"Found {len(ca1m_tar_list)} shards in {ca1m_data_dir}")

    random.shuffle(coyo_tar_list)
    random.shuffle(o365_tar_list)
    random.shuffle(general_tar_list)
    random.shuffle(ca1m_tar_list)

    # Split shards across nodes
    for tar_list in [coyo_tar_list, o365_tar_list, general_tar_list, ca1m_tar_list]:
        tar_list[:] = [str(p) for p in split_shards_by_node(
            tar_list, num_nodes=overwatch.world_size(), node_rank=overwatch.rank()
        )]

    mix_weights = variant["mix_weights"]
    overwatch.info(
        f"Mix Weights = (coyo={mix_weights['coyo_vqa']}, o365={mix_weights['o365_vqa']}, "
        f"general={mix_weights['general']}, ca1m={mix_weights['ca1m']})",
        ctx_level=1,
    )

    dataloader = build_vqa_dataloader(
        shards_coyo=coyo_tar_list,
        shards_o365=o365_tar_list,
        shards_general=general_tar_list,
        shards_ca1m=ca1m_tar_list,
        sample_transform_coyo=wild_sample_transform,
        sample_transform_o365=wild_sample_transform,
        sample_transform_general=general_sample_transform,
        sample_transform_ca1m=ca1m_sample_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        tar_shuffle=True,
        sample_shuffle=True,
        sample_shuffle_buffer=400,
        resampled=True,
        collate_fn=collate_fn,
        mix_weights=(
            mix_weights["coyo_vqa"],
            mix_weights["o365_vqa"],
            mix_weights["general"],
            mix_weights["ca1m"],
        ),
    )

    # ---- Train ----
    overwatch.info("Starting training loop")
    training_strategy.run_training_iterable(
        dataloader,
        metrics,
        save_interval=variant["save_steps"],
        start_global_step=resume_step,
        tokenizer=vlm_processor.tokenizer,
    )

    overwatch.info("Training complete. Finalizing metrics.")
    metrics.finalize()

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="HiSpatial VLM Training")
    parser.add_argument("config", type=str, help="Path to config JSON file")
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    return vars(parser.parse_args())


if __name__ == "__main__":
    faulthandler.enable()
    args = parse_args()

    configs = load_config(args["config"])
    configs["raw_config_path"] = args["config"]

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    experiment(variant=configs)
