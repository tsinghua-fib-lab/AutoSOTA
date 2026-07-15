#!/bin/bash
set -e
export PYTHONPATH=/repo
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

echo '============================================'
echo 'Stage 1: SAVAE Training'
echo '============================================'
echo "Started at: $(date)"

rm -rf /repo/svae_output
mkdir -p /repo/svae_output

python3 -u /repo/sed/svae_main.py   --config /repo/configs/vae/svae_medium.yaml   --config /repo/configs/data/sparse_scrna.yaml   --data.init_args.train_data_dir /tmp/habermann_human_lung_pf.h5ad   --data.init_args.batch_size 128   --trainer.default_root_dir /repo/svae_output   --trainer.logger.class_path lightning.pytorch.loggers.CSVLogger   --trainer.logger.init_args.save_dir /repo/svae_output   --trainer.devices 1   --trainer.enable_progress_bar false

echo "SAVAE done at: $(date)"

# Find the SVAE checkpoint
VAE_CKPT=$(find /repo/svae_output -name 'last.ckpt' | head -1)
if [ -z "$VAE_CKPT" ]; then
    echo 'ERROR: No SAVAE checkpoint found!'
    exit 1
fi
echo "SAVAE checkpoint: $VAE_CKPT"

echo ''
echo '============================================'
echo 'Stage 2: SED Training (SEDP/DDPM)'
echo '============================================'
echo "Started at: $(date)"

rm -rf /repo/sed_output
mkdir -p /repo/sed_output

python3 -u /repo/sed/sed_main.py   --config /repo/configs/sed/sed.yaml   --config /repo/configs/data/sparse_scrna.yaml   --config /repo/configs/sed/sed_unet_small.yaml   --data.init_args.train_data_dir /tmp/habermann_human_lung_pf.h5ad   --data.init_args.batch_size 128   --model.init_args.vae_dir "$VAE_CKPT"   --model.init_args.diffusion_model_config.use_ddim False   --trainer.default_root_dir /repo/sed_output   --trainer.logger.class_path lightning.pytorch.loggers.CSVLogger   --trainer.logger.init_args.save_dir /repo/sed_output   --trainer.devices 1   --trainer.enable_progress_bar false

echo "SED done at: $(date)"
echo 'All training complete!'
