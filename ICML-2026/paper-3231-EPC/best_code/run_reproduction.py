#!/usr/bin/env python3
"""Reproduction script for ePC paper (3231): FashionMNIST MLP-4 with CE loss."""
import os, sys, json, time
import lightning
import torch
import numpy as np
from datamodules import get_datamodule
from get_arch import get_architecture
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from pc_variants import get_pc_variant

os.environ["WANDB_MODE"] = "offline"

CONFIG = {
    "dataset": "FashionMNIST",
    "algorithm": "EO",
    "USE_DEEP_MLP": False,
    "USE_CROSSENTROPY_INSTEAD_OF_MSE": True,
    "e_lr": 0.003,
    "iters": 4,
    "w_lr": 5e-5,
    "max_epochs": 14,
    "batch_size": 64,
    "seeds": [0, 1, 2, 3, 4],
}


def run_seed(seed, config):
    lightning.seed_everything(seed, workers=True)

    datamodule = get_datamodule(config["dataset"], final_training_run=True)(
        config["batch_size"]
    )

    logger = CSVLogger("/autosota_artifacts/paper-3231/sota/logs", name=f"repro_ce_seed{seed}")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=config["max_epochs"],
        inference_mode=False,
        limit_predict_batches=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    architecture = get_architecture(
        dataset=datamodule.dataset_name,
        use_CELoss=config["USE_CROSSENTROPY_INSTEAD_OF_MSE"],
    )

    PC_type = get_pc_variant(
        config["algorithm"], config["USE_CROSSENTROPY_INSTEAD_OF_MSE"]
    )
    pc = PC_type(
        architecture,
        iters=config["iters"],
        e_lr=config["e_lr"],
        w_lr=config["w_lr"],
    )

    trainer.fit(pc, datamodule=datamodule)
    test_result = trainer.test(pc, datamodule=datamodule)

    pc = None
    trainer = None
    lightning.pytorch.utilities.memory.garbage_collection_cuda()
    torch.cuda.empty_cache()

    test_acc = test_result[0]["test_acc"] if test_result else None
    return test_acc


def main():
    print(f"Starting reproduction at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    cfg_display = {k: v for k, v in CONFIG.items() if k != "seeds"}
    print(f"Config: {json.dumps(cfg_display, indent=2)}")
    print(f"Seeds: {CONFIG['seeds']}")
    sys.stdout.flush()

    results = {}
    all_accs = []
    for seed in CONFIG["seeds"]:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")
        sys.stdout.flush()
        t0 = time.time()
        test_acc = run_seed(seed, CONFIG)
        elapsed = time.time() - t0
        results[f"seed_{seed}"] = test_acc
        if test_acc is not None:
            all_accs.append(test_acc)
            print(
                f"Seed {seed}: test_acc = {test_acc*100:.2f}% (elapsed: {elapsed:.0f}s)"
            )
        else:
            print(f"Seed {seed}: FAILED (elapsed: {elapsed:.0f}s)")
        sys.stdout.flush()

    if all_accs:
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        print(f"\n{'='*60}")
        print(
            f"REPRODUCTION COMPLETE: {len(all_accs)}/{len(CONFIG['seeds'])} seeds"
        )
        print(
            f"Test Accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%  "
            f"({mean_acc:.4f} +/- {std_acc:.4f})"
        )
        print(f"Individual: {[f'{a*100:.2f}' for a in all_accs]}")
        print(f"{'='*60}")

        output = {
            "paper_id": 3231,
            "condition": "CE_loss",
            "config": {k: v for k, v in CONFIG.items()},
            "results": {
                k: float(v) if v is not None else None for k, v in results.items()
            },
            "mean": float(mean_acc),
            "std": float(std_acc),
            "target_paper_value": 87.58,
            "target_ci_lower": 87.45,
            "target_ci_upper": 87.71,
        }
        with open("/repo/reproduction_results_ce.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to /repo/reproduction_results_ce.json")
    else:
        print("ERROR: No valid results!")
        sys.exit(1)


if __name__ == "__main__":
    main()
