import argparse
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random
import torch

# Allow imports from the training folder (data_module lives there)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from data_module import SequenceDataModule
from block_clt_module import CLTLightningModule

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision('medium')

def main():
    parser = argparse.ArgumentParser()
    # Path params
    parser.add_argument("--data-dir", type=str, required=True, help="Path to .a2m or .parquet file")
    parser.add_argument("--esm2-weight", type=str, required=True, help="Path to ESM2 weights .pt file")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for checkpoints/logs")

    # Model params
    parser.add_argument("--num-layers", type=int, default=12, help="Total layers in pLM (12 for ESM2 35M)")
    parser.add_argument("--d-model", type=int, default=480, help="Embedding dim (480 for ESM2 35M)")
    parser.add_argument("--d-hidden", type=int, default=4800, help="Latent dim per layer")
    parser.add_argument("--block-size", type=int, default=6,
                        help="Window size for block CLT. Layers < block_size use full accumulation; "
                             "layers >= block_size use a sliding window of this many source layers.")

    # Training params
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--auxk", type=int, default=32)
    parser.add_argument("--dead-steps-threshold", type=int, default=10000)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="ESM-BlockCLT")

    args = parser.parse_args()

    run_name = f"BlockCLT_L{args.num_layers}_D{args.d_hidden}_B{args.block_size}"
    run_output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=os.path.join(run_output_dir, "wandb")
    )

    ckpt_path = None
    last_ckpt_path = os.path.join(run_output_dir, "checkpoints", "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"Found existing checkpoint at {last_ckpt_path}. Resuming...")
        ckpt_path = last_ckpt_path

    model = CLTLightningModule(args)

    data_module = SequenceDataModule(args.data_dir, args.batch_size, num_workers=4)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_output_dir, "checkpoints"),
        filename="block_clt-{step}-{val/loss:.2f}",
        save_top_k=2,
        monitor="val/loss",
        mode="min",
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        val_check_interval=2500,
        limit_val_batches=10,
        log_every_n_steps=1,
        strategy="ddp"
    )

    trainer.validate(model, data_module, ckpt_path=ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
