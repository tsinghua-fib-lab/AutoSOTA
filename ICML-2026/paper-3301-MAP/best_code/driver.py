import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import random
import shlex
import subprocess
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb
from datasets import *

from trainers import *
from utils.constraints import *
from utils.plotting import *


def set_seed(seed):
    """Set random seeds for python, numpy and torch (CPU + CUDA).

    If seed is None, do nothing. This centralizes seeding so the CLI can
    control randomness.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    # Try to make cuDNN deterministic when possible
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    print(f"Random seed set to {seed}")


def clear_gpu_cache():
    """Clear PyTorch GPU cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def clear_all_caches():
    """Clear all available memory caches"""
    # PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Python
    gc.collect()

    # CUDA (if available)
    try:
        from numba import cuda

        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)
    except ImportError:
        pass


def monitor_memory():
    """Print current memory usage"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: Allocated = {alloc:.2f}GB, Cached = {cached:.2f}GB")

    import psutil

    ram = psutil.virtual_memory()
    print(
        f"RAM: Used = {ram.used/1024**3:.2f}GB, Available = {ram.available/1024**3:.2f}GB"
    )


def save_args(args, save_path="args.json"):
    args_dict = vars(args)
    with open(save_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    print(f"Arguments saved to {save_path}")


def _git_metadata():
    """Return best-effort git provenance without requiring git to be present."""
    metadata = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "status_short": None,
    }
    try:
        metadata["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["branch"] = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["dirty"] = bool(status)
        metadata["status_short"] = status.splitlines()
    except Exception as exc:
        metadata["error"] = str(exc)
    return metadata


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path, hash_file=True):
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return {"path": path, "exists": False}
    metadata = {
        "path": path,
        "exists": True,
        "bytes": os.path.getsize(path),
        "sha256": None,
    }
    if hash_file:
        try:
            metadata["sha256"] = _sha256_file(path)
        except Exception as exc:
            metadata["sha256_error"] = str(exc)
    return metadata


def _package_version(package_name):
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def _protein_fragments_default_path(args):
    return os.path.join(
        args.data_dir,
        "protein",
        (
            f"{args.protein_dataset_name}_fragments_"
            f"L{args.protein_fragment_length}_N{args.num_samples}.npz"
        ),
    )


def _protein_manifest_path(fragments_path):
    root, _ = os.path.splitext(fragments_path)
    return f"{root}.manifest.json"


def _load_protein_fragments(path, num_samples):
    with np.load(path) as archive:
        if "fragments" not in archive:
            raise KeyError(f"Protein fragments archive missing 'fragments': {path}")
        fragments = archive["fragments"]
    if fragments.shape[0] < num_samples:
        raise ValueError(
            f"Protein fragments archive {path} contains {fragments.shape[0]} "
            f"fragments but {num_samples} were requested."
        )
    return fragments[:num_samples]


def _dataset_artifacts(args):
    """Record known local data files that define each built-in experiment."""
    artifacts = []
    if args.problem == "bunny":
        artifacts.append(_file_metadata("data/stanford-bunny.obj"))
    elif args.problem == "smileyface_plane":
        artifacts.append(_file_metadata("data/smileyface_plane.npy"))
    elif args.problem == "smileyface_sphere":
        artifacts.append(_file_metadata("data/smileyface_sphere.npy"))
    elif args.problem == "mnist":
        mnist_raw = os.path.join(args.data_dir, "MNIST", "raw")
        for filename in (
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ):
            artifacts.append(_file_metadata(os.path.join(mnist_raw, filename)))
    elif args.problem == "protein":
        fragments_path = getattr(args, "protein_fragments_resolved_path", None)
        if not fragments_path:
            fragments_path = args.protein_fragments_path or _protein_fragments_default_path(args)
        manifest_path = _protein_manifest_path(fragments_path)
        artifacts.append(
            {
                "source": getattr(args, "protein_fragments_source", "unknown"),
                "dataset_source": "sidechainnet",
                "sidechainnet_dataset": args.protein_dataset_name,
                "sidechainnet_version": _package_version("sidechainnet"),
                "openmm_version": _package_version("openmm"),
                "fragment_length": args.protein_fragment_length,
                "requested_num_samples": args.num_samples,
                "fragments_archive": _file_metadata(fragments_path),
                "fragments_manifest": _file_metadata(manifest_path),
                "runtime_fallback_note": (
                    "If fragments_archive does not exist, driver.py fetches "
                    "SidechainNet at runtime unless --protein_fragments_path "
                    "points to a missing file, in which case it fails fast."
                ),
            }
        )
    return artifacts


def _runtime_metadata():
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "loaded_modules": os.environ.get("LOADEDMODULES", "").split(":")
        if os.environ.get("LOADEDMODULES")
        else [],
    }


def _slurm_metadata():
    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_SUBMIT_DIR",
        "SLURM_SUBMIT_HOST",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_GPUS",
        "SLURM_CPUS_ON_NODE",
        "SLURM_CLUSTER_NAME",
        "SLURM_PARTITION",
        "SLURM_ACCOUNT",
    )
    return {key: os.environ.get(key) for key in keys if os.environ.get(key)}


def _run_id_base(args):
    iso_tag = "_iso" if getattr(args, "isotropic", False) else ""
    lifted_tag = "_lifted" if getattr(args, "lifted", False) else ""
    return (
        f"{args.trainer}_samples{args.num_samples}_e{args.epochs}_nl{args.noise_level}"
        f"{lifted_tag}{iso_tag}_seed{args.seed}"
    ).replace("/", "-").replace(" ", "_")


def write_run_manifest(args, checkpoint_path, trainer_checkpoint_path=None):
    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    run_id = args.run_id or _run_id_base(args)
    timestamp_id = timestamp.replace(":", "").replace("+", "Z")
    run_dir_name = f"{timestamp_id}_{run_id}"
    run_dir = os.path.join(args.run_metadata_path, args.problem, run_id)
    if not args.run_id:
        run_dir = os.path.join(args.run_metadata_path, args.problem, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_at_utc": timestamp,
        "run_id": run_id,
        "run_dir_name": os.path.basename(run_dir),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "args": vars(args),
        "git": _git_metadata(),
        "runtime": _runtime_metadata(),
        "slurm": _slurm_metadata(),
        "data_artifacts": _dataset_artifacts(args),
        "artifacts": {
            "unified_checkpoint": _file_metadata(checkpoint_path),
            "trainer_checkpoint": _file_metadata(trainer_checkpoint_path)
            if trainer_checkpoint_path
            else None,
        },
        "notes": [
            "Checkpoint binaries are intentionally kept out of git by .gitignore.",
            "Use this manifest to match a committed run record to local models/ files or GitHub Release assets.",
        ],
    }

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Run manifest saved at {manifest_path}")
    return manifest_path


def main(args):
    save_args(
        args,
        os.path.join(
            args.model_save_path, f"args_{args.trainer}_epoch_{args.epochs}.json"
        ),
    )

    print(f"Training model with the following parameters:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    constraints_dict = None
    projector = None
    mesh = False
    image = False
    unet = False
    # Create dataset
    if args.problem == "bunny":
        dataset = BunnyDataset(
            num_samples=args.num_samples,
            mean_idx=10500,
            bunny_path="data/stanford-bunny.obj",
            mode="heat",
            noise_level=args.noise_level,
            lifted=(args.noise_level != 0.0),
        )
        constraints_dict = {"bunny": "data/stanford-bunny.obj"}
        mesh = True
        projector = MeshConstraintProjector("data/stanford-bunny.obj", device)
    elif args.problem == "smileyface_plane":
        A = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)  # Normal vetor (x-axis)
        b = torch.tensor([1.0])  # Offset (good gracious!)
        dataset = SmileyFaceDataset(
            num_samples=args.num_samples,
            A=A,
            b=b,
            lifted=True,
            noise_level=args.noise_level,
            device=device,
            isotropic=args.isotropic,
            seed=args.seed,
        )
        constraints_dict = {"linear_equality": (A, b)}
    elif args.problem == "mnist":
        # Use generic ImageFixedSumDataset with MNIST backend
        # Create a cache filename that includes num_samples, noise_level, and seed
        cache_filename = f"fixedsum_mnist_train_n{args.num_samples}_nl{args.noise_level}_lifted{int(args.lifted)}_seed{args.seed}.pt"
        cache_path = os.path.join(args.data_dir if hasattr(args, "data_dir") else "./data", "cache", cache_filename)
        
        dataset = ImageFixedSumDataset(
            device=device,
            dataset="mnist",
            pixel_sum=100.0,
            lifted=args.lifted,
            noise_level=args.noise_level,
            preload_device=device,
            train=True,
            data_root=args.data_dir if hasattr(args, "data_dir") else "./data",
            flatten=True,
            reproject_after_noise=False,
            num_samples=args.num_samples,
            random_seed=args.seed,
            cache_file=cache_path,
        )
        # Match projector target to dataset pixel_sum for consistency
        projector = FixedSumProjector(target_sum=100.0)
        image = True
    elif args.problem == "smileyface_sphere":
        sphere_center = [0.0, 0.0, 0.0]
        sphere_radius = 1.0
        dataset = SmileyFaceDataset(
            device,
            num_samples=args.num_samples,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            projection_type="sphere",
            lifted=args.lifted,
            noise_level=args.noise_level,
            isotropic=args.isotropic,
            seed=args.seed,
        )
        constraints_dict = {"sphere_equality": (sphere_center, sphere_radius)}
    elif args.problem == "MNIST_wave":
        dataset = MNISTSineManifoldDataset(
            num_samples=args.num_samples,
            digit_idx=21,
            lifted=True,
            noise_level=0.05,
            amplitude=0.05,
            frequency=2,
        )

    elif args.problem == "protein":
        protein_fragments_path = (
            args.protein_fragments_path or _protein_fragments_default_path(args)
        )
        args.protein_fragments_resolved_path = protein_fragments_path
        if os.path.exists(protein_fragments_path):
            print(f"Loading protein fragments from {protein_fragments_path}")
            fragments = _load_protein_fragments(protein_fragments_path, args.num_samples)
            args.protein_fragments_source = "local_npz"
        else:
            if args.protein_fragments_path:
                raise FileNotFoundError(
                    f"Protein fragments file not found: {args.protein_fragments_path}"
                )
            print(
                "Protein fragments archive not found at "
                f"{protein_fragments_path}; fetching SidechainNet "
                f"{args.protein_dataset_name} at runtime."
            )
            import sidechainnet as scn

            data = scn.load(name=args.protein_dataset_name, with_coordinates=True)
            fragments = extract_backbone_fragments(
                data,
                fragment_length=args.protein_fragment_length,
                max_data_length=args.num_samples,
            )
            args.protein_fragments_source = "sidechainnet_runtime"
        fragments = fragments  # placeholder for potential downsampling/scaling
        # If a CUDA device is available, precompute noisy fragments on GPU for speed.
        precompute_noise = bool(
            args.lifted and args.noise_level > 0.0 and device.type == "cuda"
        )
        # Use a reasonable batch size for precompute to avoid OOM on GPU
        precompute_batch = 128 if device.type == "cuda" else 4096
        dataset = BackboneFragmentDataset(
            fragments,
            noise_level=args.noise_level,
            lifted=args.lifted,
            device=device,
            precompute_noise=precompute_noise,
            batch_size=precompute_batch,
        )
        unet = True
        # Use the analytic fast projector for training/sampling to avoid slow
        # LBFGS-based optimization on every batch. This is a single-step
        # Gauss-Newton linearized projection (fast, approximate).
        projector = ProteinConstraintProjector(device=device)

    elif args.problem == "MoG":
        mode_params = [
            {"mean": np.array([1.0, 0.0, 0.0]), "cov": 0.01 * np.eye(3)},
            {"mean": np.array([-1.0, 1.0, 0.0]), "cov": 0.02 * np.eye(3)},
            {"mean": np.array([0.0, -1.0, 1.0]), "cov": 0.01 * np.eye(3)},
        ]
        A = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)  # Normal vetor (x-axis)
        b = torch.tensor([1.0])  # Offset (good gracious!)
        dataset = MoGDataset(
            device=torch.device("cuda"),
            num_samples=args.num_samples,
            noise_level=args.noise_level,
            A=A,
            b=b,
            mode_params=mode_params,
            lifted=args.lifted,
            isotropic=args.isotropic,
        )
        constraints_dict = {"linear_equality": (A, b)}
    elif args.problem == "custom":
        print("Not done yet! Please try something else!")
        return

    # Prefer a direct tensor path when the dataset already stores all samples
    # in memory. This avoids a slow Python-level loop over every item at startup.
    data_points = getattr(dataset, "data", None)
    if torch.is_tensor(data_points):
        data_points = data_points.clone()
    else:
        data_points = torch.stack([dataset[i] for i in range(len(dataset))])
    print(f"Dataset size: {data_points.shape}")
    # Initialize wandb (required). Fail fast if initialization fails so issues are addressed immediately.
    # Include an ISO tag for isotropic runs to keep checkpoints distinguishable
    iso_tag = "_ISO" if getattr(args, "isotropic", False) else ""
    run_name = f"{args.problem}_{args.trainer}_nl{args.noise_level}_lifted{int(args.lifted)}{iso_tag}_e{args.epochs}"
    try:
        # Increase init timeout to be robust on slow networks
        if args.no_wandb:
            wandb.init(
                project="ConstrainedDiffusionToolbox",
                name=run_name,
                config=vars(args),
                mode="disabled",
                settings=wandb.Settings(init_timeout=120),
            )
            print(f"wandb disabled for run: {run_name}")
        else:
            wandb.init(
                project="ConstrainedDiffusionToolbox",
                name=run_name,
                config=vars(args),
                settings=wandb.Settings(init_timeout=120),
            )
            print(f"wandb run initialized: {run_name}")
    except Exception as e:
        import traceback

        print("wandb.init() failed; aborting run setup.")
        print(traceback.format_exc())
        raise RuntimeError(
            "wandb.init() failed — set WANDB_API_KEY, use WANDB_MODE=offline, or pass --no_wandb"
        ) from e
    # Create trainer
    if args.trainer == "DDPM":
        trainer = DDPMTrainer(
            data_points.squeeze(),
            project_x0_sample=True,
            constraints_dict=constraints_dict if constraints_dict is not None else {},
            projector=projector if projector is not None else {},
            mesh=mesh,
            timesteps=args.num_timesteps,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            time_embed_dim=args.time_embed_dim,
            time_concat=args.time_concat,
            time_conditioning=args.time_conditioning,
            image=image,
            unet=unet,
            projector_max_iter=(
                None if args.projector_max_iter <= 0 else args.projector_max_iter
            ),
            noise_schedule=args.noise_schedule,
        )
    elif args.trainer == "DDPM_NONPROJECT":
        trainer = DDPMTrainer(
            data_points.squeeze(),
            project_x0_sample=False,
            constraints_dict=constraints_dict if constraints_dict is not None else {},
            projector=projector if projector is not None else {},
            mesh=mesh,
            timesteps=args.num_timesteps,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            time_embed_dim=args.time_embed_dim,
            time_concat=args.time_concat,
            time_conditioning=args.time_conditioning,
            image=image,
            unet=unet,
            projector_max_iter=(
                None if args.projector_max_iter <= 0 else args.projector_max_iter
            ),
            noise_schedule=args.noise_schedule,
        )
    elif args.trainer == "PIDM":
        trainer = DDPMTrainer(
            data_points.squeeze(),
            project_x0_sample=False,
            penalize_P=True,
            sample_x0=True,
            hidden_dim=args.hidden_dim,
            timesteps=args.num_timesteps,
            time_embed_dim=args.time_embed_dim,
            time_concat=args.time_concat,
            time_conditioning=args.time_conditioning,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            constraints_dict=constraints_dict if constraints_dict is not None else {},
            projector=projector if projector is not None else {},
            mesh=mesh,
            image=image,
            unet=unet,
            projector_max_iter=(
                None if args.projector_max_iter <= 0 else args.projector_max_iter
            ),
            noise_schedule=args.noise_schedule,
        )
    elif args.trainer.upper() in ("GLOW", "GLOWTRAINER", "GLOW_trainer", "Glow"):
        # GlowTrainer accepts image-like numpy data (N,H,W,C) or flat vectors
        # (N,D). Flat vectors are represented internally as 1x1 images with
        # C=D channels.
        save_dir = os.path.join(args.model_save_path, args.problem, "glow")
        os.makedirs(save_dir, exist_ok=True)
        try:
            np_data = data_points.squeeze().cpu().numpy()
        except Exception:
            # data_points may already be numpy
            try:
                np_data = data_points.squeeze().numpy()
            except Exception:
                np_data = data_points.squeeze()

        # Try to construct GlowTrainer and let it infer the shape. Keep the
        # older fallback for compatibility with external GlowTrainer variants.
        trainer = None
        try:
            trainer = GlowTrainer(np_data, image_size=None, batch_size=args.batch_size, epochs=args.epochs, save_dir=save_dir, lr=args.learning_rate, hidden_dim=args.hidden_dim)
        except Exception as e:
            # Compatibility fallback for older square-image Glow assumptions.
            try:
                if hasattr(np_data, "ndim") and np_data.ndim == 2:
                    N, D = np_data.shape
                    np_fallback = np_data.reshape(N, 1, 1, D)
                    trainer = GlowTrainer(np_fallback, image_size=1, batch_size=args.batch_size, epochs=args.epochs, save_dir=save_dir, lr=args.learning_rate, hidden_dim=args.hidden_dim)
                else:
                    raise
            except Exception:
                print(f"Failed to initialize GlowTrainer (and fallback): {e}")
                raise
    elif args.trainer.upper() in ("REALNVP", "REAL_NVP", "RNVP"):
        # Simple RealNVP trainer for vector/tabular data (or flattened images)
        save_dir = os.path.join(args.model_save_path, args.problem, "realnvp")
        os.makedirs(save_dir, exist_ok=True)
        trainer = RealNVPTrainer(
            data_points.squeeze(),
            batch_size=args.batch_size,
            lr=args.learning_rate,
            epochs=args.epochs,
            save_dir=save_dir,
            hidden_dim=args.hidden_dim,
            n_coupling_layers=6,
        )
    else:
        raise ValueError(
            f"Unknown trainer '{args.trainer}'. PDM is a sampling flag; "
            "train a DDPM_NONPROJECT checkpoint and call sample(..., PDM=True)."
        )
    trainer.train(epochs=args.epochs)

    # Ensure problem-specific model directory exists before saving
    checkpoint_dir = os.path.join(args.model_save_path, args.problem)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Include the random seed in the checkpoint filename so runs are identifiable
    # Optionally include the num_samples in the filename if requested by CLI flag.
    # Add an ISO marker to checkpoint filenames for isotropic-noise runs only
    iso_marker = "_ISO" if getattr(args, "isotropic", False) else ""
    if getattr(args, "include_num_samples_in_ckpt", False):
        checkpoint_name = (
            f"model_{args.trainer}_epoch_{args.epochs}_num_samples_{args.num_samples}_noise_level_{args.noise_level}{iso_marker}_time_{args.time_conditioning}_seed_{args.seed}.pth"
        )
    else:
        checkpoint_name = f"model_{args.trainer}_epoch_{args.epochs}_noise_level_{args.noise_level}{iso_marker}_time_{args.time_conditioning}_seed_{args.seed}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Build a robust checkpoint dict that works for trainers exposing
    # different model attribute names (denoiser, model, score_net, etc.)
    model_state = None
    for attr in ("denoiser", "model", "score_net", "score_net"):
        if hasattr(trainer, attr):
            try:
                model_obj = getattr(trainer, attr)
                model_state = model_obj.state_dict()
                break
            except Exception:
                model_state = None

    optimizer_state = None
    if hasattr(trainer, "optimizer") and getattr(trainer, "optimizer") is not None:
        try:
            optimizer_state = trainer.optimizer.state_dict()
        except Exception:
            optimizer_state = None

    # Gather common metadata if present
    training_losses = getattr(trainer, "training_losses", None)

    ckpt_dict = {}
    if model_state is not None:
        # Use a consistent key name to be compatible with existing loaders
        ckpt_dict["model_state_dict"] = model_state
    if optimizer_state is not None:
        ckpt_dict["optimizer_state_dict"] = optimizer_state
    if training_losses is not None:
        ckpt_dict["training_losses"] = training_losses

    # Add optional arrays if the trainer recorded them during training
    if hasattr(trainer, "projection_times"):
        ckpt_dict["projection_times"] = trainer.projection_times
    if hasattr(trainer, "projection_norms"):
        ckpt_dict["projection_norms"] = trainer.projection_norms
    if hasattr(trainer, "epoch_timing_breakdowns"):
        ckpt_dict["epoch_timing_breakdowns"] = trainer.epoch_timing_breakdowns

    # Save EMA (Exponential Moving Average) model weights if the trainer
    # maintains them. EMA-smoothed weights produce better sample quality
    # (Karras et al. CVPR 2024) and are preferred at evaluation time.
    if hasattr(trainer, "ema_denoiser") and trainer.ema_denoiser is not None:
        ckpt_dict["ema_model_state"] = trainer.ema_denoiser.state_dict()

    # Save noise schedule metadata for correct sampling at evaluation time.
    # The betas tensor determines the reverse-diffusion variance schedule.
    if hasattr(trainer, "noise_schedule"):
        ckpt_dict["noise_schedule"] = trainer.noise_schedule
    if hasattr(trainer, "betas"):
        ckpt_dict["betas"] = trainer.betas.cpu()

    # Save time embedding configuration for correct model architecture at eval time
    ckpt_dict["time_concat"] = args.time_concat
    ckpt_dict["time_conditioning"] = args.time_conditioning
    ckpt_dict["time_embed_dim"] = args.time_embed_dim

    # If the trainer provides a custom save_checkpoint, call it too (trainer-specific format)
    trainer_ckpt_path = None
    try:
        if hasattr(trainer, "save_checkpoint") and callable(
            getattr(trainer, "save_checkpoint")
        ):
            try:
                # Trainer-level save (keeps trainer's preferred layout)
                trainer_dir = getattr(trainer, "save_dir", checkpoint_dir)
                os.makedirs(trainer_dir, exist_ok=True)
                trainer_ckpt_path = os.path.join(trainer_dir, "checkpt.pth")
                trainer.save_checkpoint(trainer_ckpt_path)
            except Exception:
                # Non-fatal: proceed to save unified checkpoint below
                pass
    except Exception:
        pass

    # Save unified checkpoint to the central models/<problem>/ filename so
    # plotting scripts can find a consistent model_*.pth across trainers.
    try:
        torch.save(ckpt_dict, checkpoint_path)
        try:
            wandb.save(checkpoint_path)
        except Exception:
            print(
                "wandb.save() skipped because wandb is not initialized or unavailable."
            )
        print(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        print(f"Warning: failed to save unified checkpoint at {checkpoint_path}: {e}")

    try:
        write_run_manifest(args, checkpoint_path, trainer_ckpt_path)
    except Exception as e:
        print(f"Warning: failed to write run manifest: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manifold diffusion training script.")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_timesteps", type=int, default=250)
    parser.add_argument(
        "--noise_schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Noise schedule for forward diffusion: linear or cosine (Nichol & Dhariwal 2021)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_save_path", type=str, default="models/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--problem", type=str, default="bunny")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--time_embed_dim", type=int, default=32)
    parser.add_argument("--trainer", type=str, default="SBDM")
    parser.add_argument("--sampler", type=str, default="Standard")
    parser.add_argument(
        "--time_conditioning",
        type=str,
        default="default",
        choices=["default", "sinusoidal", "fourier"],
        help="Time conditioning to use for DDPM's time embedding (default: learned MLP).",
    )
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument(
        "--isotropic",
        action="store_true",
        help="If set, interpret `noise_level` as isotropic 3D Gaussian noise (instead of normal/tangent-only).",
    )
    # NOTE: Glow will infer data shape automatically from the dataset. For
    # non-image flat vectors (e.g. point clouds) we fall back to a (1,1,D)
    # image interpretation so Glow can still be used.
    parser.add_argument(
        "--time_concat",
        action="store_true",
        help="If set, use raw scalar timestep concatenation as time embedding (overrides time conditioning and time_embed_dim)",
    )
    parser.add_argument("--lifted", action="store_true")
    parser.add_argument(
        "--projector_max_iter",
        type=int,
        default=10,
        help="Max LBFGS iterations per projection (reduce to speed up training); set to 0 or -1 to leave projector default",
    )
    parser.add_argument("--num_generated_samples", type=int, default=10000)
    parser.add_argument(
        "--protein_dataset_name",
        type=str,
        default="casp12",
        help="SidechainNet dataset name for protein experiments.",
    )
    parser.add_argument(
        "--protein_fragment_length",
        type=int,
        default=10,
        help="Residue length for SidechainNet backbone fragments.",
    )
    parser.add_argument(
        "--protein_fragments_path",
        type=str,
        default=None,
        help=(
            "Optional preprocessed protein fragment .npz. If omitted, driver.py "
            "looks under data/protein/ and falls back to SidechainNet runtime fetch."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short quick test with small dataset/epochs for debugging",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb initialization/logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (int). Defaults to 42 to preserve previous behavior.",
    )
    parser.add_argument(
        "--include_num_samples_in_ckpt",
        action="store_true",
        help="If set, include the num_samples value in the saved checkpoint filename (default: False).",
    )
    parser.add_argument(
        "--run_metadata_path",
        type=str,
        default="runs",
        help="Directory for committed run manifests. Checkpoint binaries stay in model_save_path.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional explicit run identifier under run_metadata_path/<problem>/.",
    )
    args = parser.parse_args()

    # SmileyFaceDataset only applies any data noise when lifted=True.
    # Isotropic smiley-face noise is therefore also a lifted/noisy-data mode.
    if (
        args.problem in {"smileyface_plane", "smileyface_sphere"}
        and args.isotropic
        and args.noise_level > 0.0
    ):
        args.lifted = True

    os.makedirs(args.model_save_path, exist_ok=True)
    # Set random seeds early so all subsequent code uses the chosen seed
    set_seed(args.seed)
    # Quick-mode overrides to reduce startup time for development
    if args.quick:
        print(
            "Quick mode enabled: reducing num_samples, epochs, and hidden_dim for fast startup"
        )
        args.num_samples = min(args.num_samples, 1000)
        args.epochs = min(args.epochs, 2)
        args.hidden_dim = min(args.hidden_dim, 128)
        args.batch_size = min(args.batch_size, 64)
        args.time_embed_dim = min(args.time_embed_dim, 16)

    main(args)
