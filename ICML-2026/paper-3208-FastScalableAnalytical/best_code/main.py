"""
Entry-point script for generating samples with analytical diffusion models.
Multi-GPU (torchrun) version: each GPU samples a shard of num_samples in parallel.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset, DataLoader

# Local project imports
from configs.configuration import (
    ensure_run_directory,
    load_config,
    save_config,
)
from data_src import DatasetBundle, build_dataset, _build_indices_per_class
from methods import create_model, create_baseline_model
from methods.base import SamplingOutput
from services.distributed import (
    init_distributed,
    is_main,
    barrier,
    broadcast_object_from_main,
    split_count,
    RunPathsProxy,
)
from services.evaluation import (
    evaluate_main_model,
    evaluate_comparison,
    _log_metrics
)

# -----------------------------------------------------------------------------
# Global & Utils
# -----------------------------------------------------------------------------

LOGGER = logging.getLogger("local_diffusion.generate")

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate samples using locality baselines (multi-GPU supported)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a configuration file (relative to configs/ or absolute)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides in dotlist form (e.g. sampling.num_samples=8)",
    )
    return parser.parse_args(argv)

def set_random_seeds(seed: int, rank: int = 0, deterministic: bool = True) -> None:
    """Sets random seeds for reproducibility.
    
    Args:
        seed: Base random seed.
        rank: Distributed process rank (added to seed for per-rank variation).
        deterministic: If True, enables cuDNN deterministic mode for reproducible results
            (5-15% slowdown but eliminates run-to-run variance).
    """
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def setup_file_logging(log_dir: Path, rank: int = 0) -> None:
    """Configures file-based logging for the specific rank."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"generate_rank{rank:02d}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    LOGGER.info("File logging enabled at %s", log_path)

def ddp_max(value: float, *, distributed: bool, device: torch.device) -> float:
    """Reduces a value across all processes using MAX operation."""
    if not distributed:
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())

# -----------------------------------------------------------------------------
# Core Logic Helpers
# -----------------------------------------------------------------------------

def prepare_dataset(cfg: DictConfig) -> Tuple[DatasetBundle, Optional[int]]:
    """Builds the dataset and handles conditional subsetting if necessary."""
    dataset = build_dataset(cfg.dataset)
    cls_id = getattr(cfg.dataset, "cls_id", 0)

    if cfg.experiment.conditional:
        base_loader = dataset.dataloader
        torch_dataset = base_loader.dataset
        num_classes = getattr(cfg.dataset, "num_classes", 1000)

        
        # Filter for specific class indices
        indices_per_class = _build_indices_per_class(torch_dataset, num_classes=num_classes)
        cls_indices = indices_per_class[cls_id]
        cls_subset = Subset(torch_dataset, cls_indices)
        
        # Recreate DataLoader with the subset
        dataset.dataloader = DataLoader(
            cls_subset,
            batch_size=base_loader.batch_size,
            shuffle=False,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
            drop_last=False,
        )
        return dataset, cls_id
    
    return dataset, None

def run_sampling(
    model: Any,
    num_samples: int,
    batch_size: int,
    seed: int,
    rank: int,
    device: torch.device,
    return_intermediates: bool
) -> Tuple[SamplingOutput, float]:
    """Executes the model sampling loop and tracks wall time."""
    generator = torch.Generator(device=str(device))
    generator.manual_seed(int(seed) + int(rank))

    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    result = model.sample(
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        return_intermediates=return_intermediates,
    )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return result, (end_time - start_time)

def run_baseline_comparison(
    cfg: DictConfig,
    dataset: DatasetBundle,
    main_result: SamplingOutput,
    run_paths: Any,
    local_num_samples: int,
    local_batch: int,
    device: torch.device,
    rank: int,
    cls_id: Optional[int],
    return_intermediates: bool
) -> None:
    """Instantiates and runs the baseline model for comparison if configured."""
    baseline_name = cfg.get("metrics", {}).get("baseline_name")
    if not baseline_name:
        return

    baseline_path = (
        cfg.metrics.baseline_unet_path 
        if baseline_name == "baseline_unet" 
        else cfg.metrics.baseline_edm_path
    )
    
    LOGGER.info(f"[rank {rank}] Running baseline comparison against {baseline_path}")

    baseline_model = create_baseline_model(
        cfg.metrics.baseline_name,
        resolution=dataset.resolution,
        device=str(device),
        num_steps=cfg.sampling.num_inference_steps,
        model_path=baseline_path,
        dataset_name=cfg.dataset.name,
        in_channels=dataset.in_channels,
        out_channels=dataset.in_channels,
        class_id=cls_id,
    )

    generator = torch.Generator(device=str(device))
    generator.manual_seed(int(cfg.experiment.seed) + int(rank))

    baseline_result = baseline_model.sample(
        num_samples=local_num_samples,
        batch_size=local_batch,
        generator=generator,
        return_intermediates=return_intermediates,
    )

    evaluate_comparison(
        dataset,
        main_result,
        baseline_result,
        cfg,
        run_paths,
        rank=rank,
        return_intermediates=return_intermediates
    )

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    # 1. Initialization & Configuration
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    # 2. Distributed Setup
    distributed, rank, world_size, local_rank, device = init_distributed()
    if distributed and rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    
    set_random_seeds(cfg.experiment.seed, rank=rank)

    # 3. File System & Logging Setup
    run_paths = None
    if is_main(rank):
        run_paths = ensure_run_directory(cfg)
        save_config(cfg, run_paths.config)
        LOGGER.info("Run directory: %s", Path(run_paths.run_dir))

    # Broadcast run directory to other ranks
    run_dir_str = broadcast_object_from_main(
        str(run_paths.run_dir) if is_main(rank) else "", rank=rank
    )
    if not is_main(rank):
        run_paths = RunPathsProxy(Path(run_dir_str))

    if run_paths.logs is not None:
        setup_file_logging(Path(run_paths.logs), rank=rank)

    # 4. Data Preparation
    dataset, cls_id = prepare_dataset(cfg)

    # 5. Model Initialization
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    if not isinstance(model_params, dict):
        model_params = cfg.model.params # Fallback
    # ALGO-03: Pass timestep_schedule from sampling config to model
    if hasattr(cfg.sampling, 'timestep_schedule'):
        model_params['timestep_schedule'] = cfg.sampling.timestep_schedule

    model = create_model(
        cfg.model.name,
        dataset=dataset,
        device=str(device),
        num_steps=cfg.sampling.num_inference_steps,
        params=model_params,
        class_id=cls_id,
    )
    model.train(dataset) # Load/Train dataset-dependent stats

    # 6. Compute Batch Sizes & Shards
    global_num_samples = int(cfg.sampling.num_samples)
    local_num_samples, sample_offset = split_count(global_num_samples, rank, world_size)
    
    global_batch = int(cfg.sampling.batch_size)
    if distributed and global_batch % world_size != 0:
        raise ValueError(
            f"sampling.batch_size ({global_batch}) must be divisible by world_size ({world_size})"
        )
    local_batch = global_batch // world_size if distributed else global_batch
    
    # 7. Sampling Execution
    return_intermediates = (
        cfg.metrics.output.save_intermediate_images or 
        cfg.metrics.baseline_path is not None
    )

    result, local_time = run_sampling(
        model=model,
        num_samples=local_num_samples,
        batch_size=local_batch,
        seed=cfg.experiment.seed,
        rank=rank,
        device=device,
        return_intermediates=return_intermediates
    )

    # 7b. Write profiling summary (CODE-01)
    if is_main(rank) and hasattr(model, 'get_profile_summary'):
        profile_summary = model.get_profile_summary()
        if profile_summary.get('profile_enabled'):
            import json
            profile_path = Path(run_paths.run_dir) / 'profile_metrics.json'
            profile_path.write_text(json.dumps(profile_summary, indent=2))
            LOGGER.info("Profile summary written to %s", profile_path)

    # 8. Metrics & Evaluation
    sampling_time_total = ddp_max(local_time, distributed=distributed, device=device)

    evaluate_main_model(
        model,
        dataset,
        result,
        cfg,
        run_paths,
        sampling_time_total=sampling_time_total,
        rank=rank,
        world_size=world_size,
        sample_offset=sample_offset,
        local_batch=local_batch,
    )

    # 9. Baseline Comparison (Optional)
    run_baseline_comparison(
        cfg=cfg,
        dataset=dataset,
        main_result=result,
        run_paths=run_paths,
        local_num_samples=local_num_samples,
        local_batch=local_batch,
        device=device,
        rank=rank,
        cls_id=cls_id,
        return_intermediates=return_intermediates
    )

    # 10. Cleanup
    barrier()
    if distributed and is_main(rank):
        LOGGER.info("All ranks finished.")
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()