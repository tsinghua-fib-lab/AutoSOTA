"""
Run certified unlearning experiment locally without wandb.
Prints per-epoch validation accuracy and computes epochs-to-threshold metrics.
"""
import logging
import os
import sys
import json
import time
from unittest.mock import MagicMock, patch

# === Mock wandb ===
import types
wandb_mock = types.ModuleType("wandb")
wandb_mock.init = lambda **kwargs: MagicMock()
wandb_mock.log = lambda *args, **kwargs: None
wandb_mock.run = MagicMock()
wandb_mock.run.id = "local_run"
wandb_mock.finish = lambda: None
wandb_mock.config = MagicMock()
sys.modules["wandb"] = wandb_mock

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from src.data.datamodule import get_datamodule
from src.models.model import ModelFactory
from src.training.trainer import Trainer
from src.utils.utils import load_config, logger, parse_args

LOGS_DIR = "logs/local_run"

def main():
    print(f"JAX devices: {jax.local_devices()}", flush=True)

    args = parse_args()
    config = load_config(args.config)

    run_dir = LOGS_DIR
    os.makedirs(run_dir, exist_ok=True)
    import shutil
    shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    checkpoint_path = os.path.join(run_dir, "ckpt")
    os.makedirs(checkpoint_path, exist_ok=True)
    config["checkpoint_path"] = checkpoint_path

    data_module = get_datamodule(config=config, generator=None)
    data_module.setup()

    model = ModelFactory.create_model(
        model_name=config["model"]["name"],
        num_classes=config["model"]["n_classes"],
    )
    key = jax.random.PRNGKey(config["global_seed"])

    trainer = Trainer(
        config=config,
        model=model,
        key=key,
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        test_dataloader=data_module.test_dataloader(),
        forget_dataloader=data_module.forget_dataloader(),
        retain_dataloader=data_module.retain_dataloader(),
        use_pretrained=config["training"]["pretrained"],
        run_dir=run_dir,
    )

    # Track per-epoch accuracy
    val_acc_history = {}
    original_validate = trainer.validate

    def patched_validate(state, data_loader, epoch, prefix):
        val_metrics = trainer.evaluate(state, data_loader, prefix)
        epoch_prefix = "epoch" if "train" in prefix.lower() else "unlearn_epoch"
        val_metrics[epoch_prefix] = epoch
        acc_key = f"{prefix}_accuracy"
        if acc_key in val_metrics:
            acc = float(val_metrics[acc_key]) * 100.0
            print(f"{prefix} Epoch {epoch}: accuracy={acc:.2f}%", flush=True)
            if "Unlearn" in prefix and "Val" in prefix:
                val_acc_history[epoch] = acc
        logger.info(val_metrics)

    trainer.validate = patched_validate

    state = trainer.fit()
    state = trainer.unlearn(state)
    trainer.test(state=state)

    # Compute epochs-to-threshold for val accuracy
    thresholds = [30, 35, 40, 45, 50]
    results = {}
    for thr in thresholds:
        met_at = None
        for ep in sorted(val_acc_history.keys()):
            if val_acc_history[ep] >= thr:
                met_at = ep
                break
        results[f"epochs_to_{thr}pct_val_acc"] = met_at if met_at is not None else -1

    print("\n=== EVAL RESULTS ===", flush=True)
    for k, v in results.items():
        print(f"{k}: {v}", flush=True)
    print("=== END RESULTS ===", flush=True)

    # Also dump JSON for easy parsing
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f)
    print(f"Results saved to {run_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
