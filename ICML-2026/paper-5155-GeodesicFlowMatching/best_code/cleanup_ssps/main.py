#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from cleanup_ssps.space_factory import build_ssp_space, resolve_encoded_dim
from cleanup_ssps.run import FeedforwardTrainer, FlowTrainer
from cleanup_ssps.model import ResidualMLP
from utils.training import TrainingManager
from utils.evaluation import EvaluationManager
from utils.wandb_utils import initialize_wandb
from utils.config_loader import load_experiments
from cleanup_ssps.dataset_registry import DatasetSpec, ensure_target_dataset

# -----------------------------------------------------------------------------
# Absolute paths
# -----------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "experiments.yaml"
SAVE_ROOT    = PROJECT_ROOT / "trained_models"

def drift_filename(mode: str) -> str:
    # feedforward stored as "feedforward.pt", others as "drift_{mode}.pt"
    return "feedforward.pt" if mode.lower() in ("ff", "feedforward") else f"drift_{mode}.pt"

def score_filename(mode: str) -> str:
    return f"score_{mode}.pt"


def _extract_modules_from_training_entry(entry):
    """Unpack TrainingManager.train() value: (models, train_losses, val_losses)."""
    models_part, _, _ = entry
    if isinstance(models_part, (tuple, list)):
        return [m for m in models_part if isinstance(m, nn.Module)]
    if isinstance(models_part, nn.Module):
        return [models_part]
    raise TypeError(f"Unexpected models payload: {type(models_part)}")


def load_or_train_mode(
    mode: str,
    encoded_dim: int,
    save_dir: Path,
    ssp_space,
    trainer_config: dict,
    ssp_config: dict,
    device: str
) -> list[nn.Module]:
    """
    Returns [drift_module] or [drift_module, score_module].
    If the checkpoint exists in save_dir, loads it; otherwise trains that mode.
    """
    ckpt_d = save_dir / drift_filename(mode)
    ckpt_s = save_dir / score_filename(mode)

    # 1) Load from checkpoint if it exists
    if ckpt_d.exists():
        print(f"  → Loading cached [{mode}]")
        if mode.lower() in ("ff", "feedforward"):
            net = ResidualMLP(encoded_dim, flow=False).to(device)
            net.load_state_dict(torch.load(ckpt_d, map_location=device))
            return [net]
        else:
            net = ResidualMLP(encoded_dim, flow=True)
            trainer = FlowTrainer(
                encoded_dim    = encoded_dim,
                architecture   = net,
                data_dir       = trainer_config["data_dir"],
                batch_size     = trainer_config["batch_size"],
                epochs         = trainer_config["epochs"],
                lr             = trainer_config["lr"],
                weight_decay   = trainer_config.get("weight_decay", 0.0),
                val_split      = trainer_config.get("val_split", 0.1),
                signal_strength= trainer_config.get("signal_strength", 0.0),
                noise_type     = trainer_config.get("noise_type", "uniform_hypersphere"),
                target_type    = trainer_config.get("target_type", "coordinate"),
                device         = device,
                sampling_mode  = mode
            )
            fm = trainer.flow_model
            fm.model.load_state_dict(torch.load(ckpt_d, map_location=device))
            modules = [fm.model]
            if ckpt_s.exists():
                fm.score_model.load_state_dict(torch.load(ckpt_s, map_location=device))
                modules.append(fm.score_model)
            return modules

    # 2) Otherwise train via TrainingManager
    print(f"  → No checkpoint for [{mode}] — training it now")
    tc = dict(trainer_config)
    mode_lower = mode.lower()
    if mode_lower in ("ff", "feedforward"):
        tc["train_feedforward"] = True
        tc["sampling_modes"] = []
    else:
        tc["train_feedforward"] = False
        tc["sampling_modes"] = [mode]

    tm = TrainingManager(ssp_space, tc, ssp_config)
    results = tm.train()

    if mode_lower in ("ff", "feedforward"):
        matching = [k for k in results if k[0].endswith("_FF")]
    else:
        matching = [k for k in results if k[1] == mode]
    if not matching:
        raise KeyError(f"No training result for mode '{mode}'. Available keys: {list(results.keys())}")
    key = matching[0]
    entry = results.pop(key)
    modules = _extract_modules_from_training_entry(entry)

    save_dir.mkdir(parents=True, exist_ok=True)
    drift_mod = modules[0]
    torch.save(drift_mod.state_dict(), ckpt_d)
    print(f"    • Saved drift [{mode}] → {ckpt_d}")
    if len(modules) > 1:
        score_mod = modules[1]
        torch.save(score_mod.state_dict(), ckpt_s)
        print(f"    • Saved score [{mode}] → {ckpt_s}")

    return modules

def main():
    experiments = load_experiments(str(CONFIG_PATH))

    for experiment in experiments:
        ssp_cfg = experiment["ssp_config"]
        tr_cfg  = experiment["trainer_config"]
        modes   = list(tr_cfg["sampling_modes"])

        # compute encoded_dim from explicit config or bundle params
        enc_dim = resolve_encoded_dim(ssp_cfg)
        ssp_cfg["encoded_dim"] = enc_dim

        # initialize wandb
        initialize_wandb(
            project_name    = "Clean_Up",
            experiment_name = experiment["name"],
            tags            = experiment["tags"],
            config          = {**ssp_cfg, **tr_cfg}
        )

        # choose device
        device = tr_cfg["device"]
        if device == "cuda" and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("Using CPU")
            device = "cpu"

        # build SSP space from bundle class config
        t0 = time.time()
        domain_dim = int(ssp_cfg.get("domain_dim", 2))
        domain_bounds = np.asarray(
            ssp_cfg.get("domain_bounds", [[-1, 1], [-1, 1]]),
            dtype=float,
        )
        ssp_space = build_ssp_space(
            ssp_cfg,
            domain_dim=domain_dim,
            domain_bounds=domain_bounds,
        )
        print(f"SSP space ({type(ssp_space).__name__}) created in {(time.time() - t0):.2f}s")

        # ensure target dataset exists for this exact SSP configuration
        data_root = PROJECT_ROOT / tr_cfg.get("data_root", "data")
        dataset_spec = DatasetSpec(
            data_root=data_root,
            dataset_type=tr_cfg.get("target_dataset_type", "coordinate_ssps"),
            encoded_dim=enc_dim,
            length_scale=float(ssp_cfg["length_scale"]),
            train_samples=int(tr_cfg.get("train_samples", 20000)),
            test_samples=int(tr_cfg.get("test_samples", 5000)),
            sampling_method=tr_cfg.get("dataset_sampling_method", "sobol"),
            train_subdir=tr_cfg.get("train_subdir", "train"),
            test_subdir=tr_cfg.get("test_subdir", "test"),
        )
        dataset_info = ensure_target_dataset(dataset_spec, ssp_cfg, ssp_space)
        tr_cfg["data_dir"] = dataset_info["train_dir"]
        tr_cfg["test_dir"] = dataset_info["test_dir"]
        state = "created" if dataset_info["created"] else "reused"
        grp = dataset_info.get("dataset_group", "?")
        print(f"Dataset {state}: group={grp} id={dataset_info['dataset_id']}")

        # prepare save directory (same naming convention as dataset geometry folder)
        run_folder = f"dim{enc_dim}_ls{ssp_cfg['length_scale']}"
        save_dir = SAVE_ROOT / dataset_info.get("dataset_group", run_folder)
        save_dir.mkdir(parents=True, exist_ok=True)

        # train or load each mode
        training_results: dict[tuple[str,str], list[nn.Module]] = {}
        print(f"→ Modes to train/load: {modes}")
        for mode in modes:
            training_results[(mode, mode)] = load_or_train_mode(
                mode, enc_dim, save_dir,
                ssp_space, tr_cfg, ssp_cfg, device
            )

        # evaluation
        t1         = time.time()
        eval_cfg   = experiment.get("eval_config", {})
        noise_lvls = eval_cfg.get("signal_strengths", [0.0, 0.25, 0.5, 0.75, 1.0])
        eval_steps = eval_cfg.get("num_steps",    [1, 2, 5, 10, 50])
        repeats    = eval_cfg.get("repeats", 5)

        eval_mgr = EvaluationManager(
            training_results,
            test_dir         = tr_cfg["test_dir"],
            device           = device,
            signal_strengths = noise_lvls,
            eval_steps       = eval_steps,
            repeats          = repeats,
            noise_type       = tr_cfg.get("noise_type", "uniform_hypersphere"),
            target_type      = tr_cfg.get("target_type", "coordinate"),
        )
        eval_mgr.run_all(ssp_space, batch_size=tr_cfg["batch_size"])
        print(f"Evaluation completed in {(time.time() - t1):.2f}s\n")

if __name__ == "__main__":
    main()
