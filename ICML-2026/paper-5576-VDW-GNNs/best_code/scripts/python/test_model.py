#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime
import torch
from accelerate import Accelerator
from config.arg_parsing import get_clargs
from config.config_manager import ConfigManager
from training.prep_dataset import load_dataset
# from training.prep_dataloaders import create_dataloaders
from training.prep_model import prepare_vdw_model
from data_processing.data_utilities import get_random_splits
import models.nn_utilities as nnu
from scripts.python.main_training import run_evaluation

DEFAULT_SNAPSHOT_NAME = 'best'

def find_newest_config_file(config_dir):
    config_files = [
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith('.yaml') or f.endswith('.yml')
    ]
    if not config_files:
        return None
    # Sort by modification time, newest last
    config_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return config_files[0]

def main():
    # Parse command-line arguments
    clargs = get_clargs()
    if not hasattr(clargs, 'experiment_dir') or clargs.experiment_dir is None:
        print("Error: --experiment_dir argument is required for testing.")
        sys.exit(1)

    exp_dir = clargs.experiment_dir
    config_dir = os.path.join(exp_dir, 'config')

    # Determine config_path: use provided, or find newest in config_dir
    config_path = getattr(clargs, 'config_path', None)
    if not config_path:
        config_path = find_newest_config_file(config_dir)
        if not config_path:
            print(f"Error: No config YAML files found in {config_dir}")
            sys.exit(1)
        print(f"[INFO] Automatically selected newest config in '{config_dir}': {config_path}")
    clargs.config_path = config_path

    # Set snapshot_name (default: DEFAULT_SNAPSHOT_NAME) and model_save_dir to exp_dir/models
    snapshot_name = getattr(clargs, 'snapshot_name', None) or DEFAULT_SNAPSHOT_NAME
    clargs.snapshot_name = snapshot_name
    clargs.model_save_dir = os.path.join(exp_dir, 'models')

    # Load configuration
    config_manager = ConfigManager(clargs)
    config = config_manager.config
    config.snapshot_name = snapshot_name
    config.model_save_dir = os.path.join(exp_dir, 'models')
    config.train_logs_save_dir = os.path.join(exp_dir, 'logs')
    config.results_save_dir = os.path.join(exp_dir, 'metrics')

    # ------------------------------------------------------------------
    # Patch: ensure target preprocessing statistics are present so that the
    # model registers _target_center / _target_scale buffers.  If these were
    # computed during training but never written to the YAML config, the
    # statistics will be absent here, which would cause a mismatch when
    # loading the snapshot ("Unexpected key(s) in state_dict: ...").
    # We therefore insert dummy statistics that will be overwritten by
    # the real values coming from the checkpoint.
    # ------------------------------------------------------------------
    # Ensure target preprocessing statistics placeholders exist *only if* they are missing.
    # We avoid overwriting valid statistics already present in the YAML.
    if not getattr(config.dataset_config, 'target_preproc_stats', None):
        target_dim = getattr(config.dataset_config, 'target_dim', 1)
        config.dataset_config.target_preproc_stats = {
            'center': None,  # will be overwritten by checkpoint values
            'scale' : None,
        }

    # Initialize accelerator
    acc = Accelerator(
        device_placement=False if config.using_pytorch_geo else config.device,
        mixed_precision='no' if config.mixed_precision == 'none' else config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        split_batches=config.dataloader_split_batches,
    )

    # Print accelerator info
    if acc.is_main_process:
        print(f"Accelerator info:")
        print(f"   - num. processes: {acc.num_processes}")
        print(f"   - distributed type: {acc.distributed_type}")

    # Load dataset
    dataset = load_dataset(
        config, 
        model_key=config.model_config.model_key
    )
    splits_dict = get_random_splits(
        n=len(dataset),
        seed=config.dataset_config.split_seed,
        train_prop=config.dataset_config.train_prop,
        valid_prop=config.dataset_config.valid_prop
    )
    test_indices = splits_dict.get('test')
    set_name = 'test'
    if test_indices is None:
        test_indices = splits_dict.get('valid')
        set_name = 'valid'
    if test_indices is None:
        raise ValueError("No test or valid set found in splits_dict for evaluation.")
    from torch.utils.data import Subset
    test_dataset = Subset(dataset, test_indices)
    
    # Create only the test/valid dataloader
    if config.using_pytorch_geo:
        from torch_geometric.data import DataLoader as PyGDataLoader
        test_loader = PyGDataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=getattr(config, 'dataloader_num_workers', config.num_workers),
            pin_memory=config.pin_memory,
            drop_last=config.drop_last
        )
    else:
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=getattr(config, 'dataloader_num_workers', config.num_workers),
            pin_memory=config.pin_memory,
            drop_last=config.drop_last
        )
    dataloader_dict = {set_name: test_loader}

    # Prepare model
    key_lower = config.model_config.model_key.lower()
    if key_lower in ('vdw', 'vdw_radial'):
        # prepare_vdw_model internally handles both standard and radial variants
        config, model, dataloader_dict = prepare_vdw_model(
            config, dataloader_dict, acc=acc
        )
    else:
        raise ValueError(f"Model key '{config.model_config.model_key}' not supported")

    # Run a dummy forward pass, create optimizer/scheduler, and register checkpoints in correct order
    # (mimic snapshot loading logic from train.py)
    model = model.to(acc.device)
    model = acc.prepare(model)
    if isinstance(dataloader_dict, dict):
        dataloader_dict[set_name] = acc.prepare(
            dataloader_dict[set_name]
        )
    elif hasattr(dataloader_dict, 'to'):
        dataloader_dict = acc.prepare(dataloader_dict)
    # Run dummy forward pass to initialize submodules
    model.eval()
    with torch.no_grad():
        dummy_batch = next(iter(dataloader_dict[set_name]))
        if hasattr(dummy_batch, 'to'):
            dummy_batch = dummy_batch.to(acc.device)
        elif isinstance(dummy_batch, (tuple, list)):
            dummy_batch = [x.to(acc.device) for x in dummy_batch]
        model(dummy_batch)

    # -------------------------------------------------
    # Load the model weights; ignore optimizer / scheduler checkpoints
    # -------------------------------------------------
    # Prefer snapshot name, then 'best', then 'final' model weights
    candidates = [
        os.path.join(config.model_save_dir, snapshot_name),
        os.path.join(config.model_save_dir, 'best', snapshot_name),
        os.path.join(config.model_save_dir, 'final', snapshot_name),
    ]
    snapshot_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if snapshot_dir is None:
        raise FileNotFoundError(f"Snapshot directory for '{snapshot_name}' not found in any expected location.")
    print(f"[INFO] Loading model weights from: {snapshot_dir}")

    # Prefer safetensors but fall back to a .bin file for older runs
    safetensor_path = os.path.join(snapshot_dir, 'model.safetensors')
    bin_path = os.path.join(snapshot_dir, 'pytorch_model.bin')
    if os.path.isfile(safetensor_path):
        try:
            from safetensors.torch import load_file as safe_load
        except ImportError:
            raise ImportError("safetensors is not installed. Install it with `pip install safetensors`. ")
        # Load tensors onto CPU; they will be moved to the correct device when
        # load_state_dict copies them into the (GPU/TPU) model.
        state_dict = safe_load(safetensor_path, device="cpu")
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location=acc.device)
    else:
        raise FileNotFoundError("Neither 'model.safetensors' nor 'pytorch_model.bin' found in snapshot directory.")

    # Load into the (possibly DDP-wrapped) model
    unwrapped_model = acc.unwrap_model(model)
    missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)
    if acc.is_main_process and getattr(config, 'verbosity', 0) > -1:
        print(f"[INFO] Weights loaded (missing={len(missing)}, unexpected={len(unexpected)}) from {os.path.basename(safetensor_path if os.path.isfile(safetensor_path) else bin_path)}")
    acc.wait_for_everyone()

    # If needed, (a) subset the targets to only those specified by target_include_indices,
    # (b) normalize targets using stats stored with the model (these stats are attributes of BaseModule)
    target_incl_idx = torch.tensor(config.dataset_config.target_include_indices, dtype=torch.long)
    with torch.no_grad():
        center = unwrapped_model._target_center.cpu()
        scale  = unwrapped_model._target_scale.cpu()
    if (target_incl_idx is not None) \
    or ((center is not None) and (scale is not None)):
        for data in test_dataset:
            if target_incl_idx is not None:
                y = data.y.squeeze()[target_incl_idx]
            if (center is not None) and (scale is not None):
                y = (y - center) / scale
            # Ensure y is at least 1-D for later tensor operations
            if y.dim() == 0:
                y = y.unsqueeze(0)
            data.y = y

    # Run evaluation
    run_evaluation(
        config=config,
        dataloader_dict=dataloader_dict,
        trained_model=model,
        accelerator=acc,
        fold_idx=0,
        fold_times_l=None,
        eval_set_key=set_name,
        metrics_kwargs={
            'num_outputs': config.dataset_config.target_dim,
            'target_include_indices': getattr(config.dataset_config, 'target_include_indices', None),
        },
        eval_start_time=time.time()
    )

if __name__ == '__main__':
    main() 

    