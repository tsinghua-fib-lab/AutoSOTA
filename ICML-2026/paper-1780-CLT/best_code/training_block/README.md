# Training

This folder contains the training code for the windowed Cross-Layer Transcoder (CLT) model.

## Key Components

- `block_clt_model.py`: Windowed CLT model architecture with top-k sparse activation and cross-layer decoding
- `block_clt_module.py`: PyTorch Lightning training module with loss functions and optimization
- `run_block_clt.py`: Main training script with argument parsing and logging
- `main.sh`: Shell script for running training with default parameters

## Usage

Run training with default settings:

```bash
./main.sh
```

Or customize parameters by changing variables in main.sh. You can also find the trained windowed CLT for ESM2-35M
at https://huggingface.co/ktalreja/ProtoMechModels/tree/main/BlockCLT_L12_D4800_B4