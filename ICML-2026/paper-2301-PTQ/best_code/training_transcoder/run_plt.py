import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.append("../training") 
from data_module import SequenceDataModule 
from plt_module import PLTLightningModule
import numpy as np
import random
import torch
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
    parser.add_argument("--output-dir", type=str, default="results_plt", help="Directory for checkpoints/logs")
    
    # Model params
    parser.add_argument("--num-layers", type=int, default=6, help="Total layers in pLM")
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--d-hidden", type=int, default=3200, help="Latent dim per layer")
    
    # Training params
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--auxk", type=int, default=32)
    parser.add_argument("--dead-steps-threshold", type=int, default=10000)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="ESM-PLT")
    
    args = parser.parse_args()
    
    # Create output directory
    run_name = f"PLT_L{args.num_layers}_D{args.d_hidden}"
    run_output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Logger
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
    
    # Model
    model = PLTLightningModule(args)
    
    # Data
    data_module = SequenceDataModule(args.data_dir, args.batch_size, num_workers=4)
    
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_output_dir, "checkpoints"),
        filename="plt-{step}-{val/loss:.2f}",
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