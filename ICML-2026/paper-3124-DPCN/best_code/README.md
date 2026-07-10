# DeepPCNs

This repository contains the official implementation of the ICML 2026 paper "Towards the Training of Deeper Predictive Coding Neural Networks".

## Requirements

- Python 3.10+
- JAX
- [pcx](https://github.com/changqi97/pcx)
- optax
- torch, torchvision (for data loading only)
- omegacli / omegaconf
- scikit-learn (for Galaxy10 dataset)
- h5py (for Galaxy10 dataset)

## File Structure

```
├── dataset.py          # Dataset loaders and utilities
├── model.py            # VGG and ResNet model definitions for PC
├── model_BN.py         # VGG and ResNet model definitions with BatchNorm
├── PC.py               # Predictive Coding (PC) training
├── iPC.py              # Incremental Predictive Coding (iPC) training
├── PC_BN.py            # Predictive Coding (PC) training with BatchNorm
├── iPC_BN.py           # Incremental Predictive Coding (iPC) training with BatchNorm
├── seed.py             # Multi-seed training runner and result logger
└── README.md
```

### File Descriptions

- **dataset.py**: Data loaders for MNIST, FashionMNIST, CIFAR-10, CIFAR-100, TinyImageNet, ImageNet-32, ImageNet-10, ImageNette, and Galaxy10. Update `DATA_ROOT` and `DATA_ROOT_SHARED` at the top of the file to match your data paths.
- **model.py**: Model architectures including VGG5, VGG7, VGGNet (VGG9-VGG19), ResNet10, and ResNet18, all implemented as Predictive Coding energy-based modules. Precision schedule types:
  - `S` (Spiking Precision): Binary spiking schedule where each layer activates at a specific time step
  - `D` (Decaying Precision): Logarithmically decaying schedule per layer
  - `PC` / `BP`: Uniform precision (all weights = 1.0)
- **model_BN.py**: Same architectures as `model.py` but with `BatchNorm` layers added after each convolution. During inference, BatchNorm state is controlled via `init_step` and `inference` flags to ensure running statistics are only updated during the weight update phase.
- **PC.py**: Standard Predictive Coding training. Supports two weight update modes controlled by `Forward_type`:
  - `FU` (Forward Update): Weight gradient computed from the initial forward pass (h0)
  - `PC`: Weight gradient computed from the settled hidden states (h)
- **iPC.py**: Incremental Predictive Coding training. Weights are updated at every inference step using a stoppable optimizer that skips updates when gradients are zero.
- **PC_BN.py**: Standard Predictive Coding training with BatchNorm. Same as `PC.py` but uses BatchNorm models from `model_BN.py`. 
- **iPC_BN.py**: Incremental Predictive Coding training with BatchNorm. Same as `iPC.py` but uses BatchNorm models from `model_BN.py`. 
- **seed.py**: Manages multi-seed training runs (default: 5 seeds). Saves intermediate results every 10 epochs and after each seed completes.

## Usage

### Training

Run PC training:

```bash
python PC.py <config.yaml>
```

Run iPC training:

```bash
python iPC.py <config.yaml>
```

Run PC training with BatchNorm:

```bash
python PC_BN.py <config.yaml>
```

Run iPC training with BatchNorm:

```bash
python iPC_BN.py <config.yaml>
```

### Configuration

Training is configured via YAML files. 

```yaml
hp:
  T: 13                          # Number of inference steps
  act_fn: gelu                  # Activation function
  alpha: 0.001                  # Precision schedule base value
  batch_size: 128
  dataset: CIFAR10              # Dataset name
  epochs: 30
  model: ResNet10                # Model name (ResNet10, ResNet18, VGG5, VGG7, VGG10, etc.)
  optim:
    w:                          # Weight optimizer (AdamW)
      lr: 0.00075
      wd: 0.00003
    x:                          # Hidden state optimizer (SGD with momentum)
      lr: 0.56
      momentum: 0.9
      nesterov: true
  precision_type: S             # Precision schedule type: S, D, PC/BP
  Forward_type: FU              # Weight update mode (PC.py only): FU or PC
  se_flag: true                 # Use squared error (true) or cross-entropy (false)
```

### Key Hyperparameters


| Parameter        | Values              | Description                                  |
| ---------------- | ------------------- | -------------------------------------------- |
| `precision_type` | `S`, `D`, `PC`/`BP` | Precision schedule for inference             |
| `Forward_type`   | `FU`, `PC`          | Weight update energy function (PC.py only)   |
| `T`              | int                 | Number of inference steps per batch          |
| `se_flag`        | `true`/`false`      | Squared error vs cross-entropy output energy |


### Supported Datasets

MNIST, FashionMNIST, CIFAR10, CIFAR100, TinyImageNet, ImageNet32, ImageNet, Galaxy10, ImageNette

### Supported Models

VGG5, VGG7, VGG9, VGG10, VGG11, VGG13, VGG15, VGG19, ResNet10, ResNet18

### Output

Results are saved incrementally to `<config_name>_accuracy.json`:

- Every 10 epochs: current seed's progress is saved
- After each seed completes: aggregated results across completed seeds are saved
- After all seeds complete: final results

```json
{
    "status": "running",
    "completed_seeds": 3,
    "total_seeds": 5,
    "best_per_seed": [0.92, 0.91, 0.93],
    "avg": 0.92,
    "std": 0.01,
    "seed_0": {"epoch": 30, "best_accuracy": 0.92, "accuracies": [...]},
    ...
}
```

## Citation
If you found this library to be useful in your work, then please cite: [DeepPCNs](https://arxiv.org/abs/2506.23800), [pcx](https://arxiv.org/abs/2407.01163)

```bibtex
@inproceedings{qi2026towards,
      title={Towards the Training of Deeper Predictive Coding Neural Networks},
      author={Chang Qi and Matteo Forasassi and Thomas Lukasiewicz and Tommaso Salvatori},
      booktitle={Forty-third International Conference on Machine Learning},
      year={2026},
      url={https://arxiv.org/abs/2506.23800}
}
```

```bibtex
@article{pinchetti2024benchmarkingpredictivecodingnetworks,
      title={Benchmarking Predictive Coding Networks -- Made Simple}, 
      author={Luca Pinchetti and Chang Qi and Oleh Lokshyn and Gaspard Olivers and Cornelius Emde and Mufeng Tang and Amine M'Charrak and Simon Frieder and Bayar Menzat and Rafal Bogacz and Thomas Lukasiewicz and Tommaso Salvatori},
      year={2024},
      eprint={2407.01163},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01163}, 
}
```

## Notes

- Update data paths in `dataset.py` (`DATA_ROOT` and `DATA_ROOT_SHARED`) before running.
- The `seed.py` file uses `SEEDS = [0, 1, 2, 3, 4]` by default. Modify as needed.
- Model weights are saved to `./weights/` when training with seed 0.

