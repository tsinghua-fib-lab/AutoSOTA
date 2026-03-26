# Certified Unlearning for Neural Networks
This repository contains the implementation for the paper "Certified Unlearning for Neural Networks" (ICML 2025).

## Methods
- **PABI (Privacy Amplification by Iteration)**: `Gradient Clipping` in the paper
- **DP-SGD**: With group privacy as described in the paper
- **Contractive Coefficients**: `Model Clipping` in the paper
- **DP-Baseline**: `Output Perturbation` in the paper
- **Retrain**


## Installation

```bash
# Clone the repository
git clone https://github.com/stair-lab/certified-unlearning-neural-networks-icml-2025.git
cd certified-unlearning-neural-networks-icml-2025

# alternatively use conda
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```


## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── base_datamodule.py     # Base data loading functionality
│   │   ├── cifar_dataset.py       # CIFAR-10/100 data loaders
│   │   ├── mnist_dataset.py       # MNIST data loader
│   │   └── datamodule.py          # Data module factory
│   ├── models/
│   │   ├── model.py               # Model factory
│   │   ├── tiny_net.py             
│   │   └── two_layer_net.py        
│   ├── training/
│   │   └── trainer.py             # Main training/unlearning logic
│   └── utils/
│       ├── dp.py                  
│       └── utils.py               
├── experiment.py                  # Main experiment runner
├── run_exp.py                     # Multi-GPU experiment launcher
├── feature_extractor.py           # Extract ResNet-18 backbone features 
```
## Experiment Configurations

The repository includes pre-configured experiments in the following directories:
- `budget_curve_runs/`: Privacy budget analysis experiments   - Figure 1
- `convergence_curve_runs/`: Convergence analysis experiments - Figure 2
- `dp_sgd_runs/`: DP-SGD and baseline experiments             - Figure 3
- `eps_sweep_runs/`: Privacy parameter (ε) sweep experiments  - Figure 4 (in Appendix)



## Usage

### Running a Single Experiment

To run a single experiment with a specific configuration:

```bash
python experiment.py --config dp_sgd_runs/dp-sgd-cifar10-fc/1tmh6p2f_config.yaml
```
### Running Multiple Experiments

To run multiple experiments in parallel across GPUs:

```bash
python run_exp.py -n <num_gpus> -e <experiment_dir> [--offset <gpu_offset>] [-j <jobs_per_gpu>]
```
Parameters:
- `-n`: Number of GPUs to use
- `-e`: Directory containing experiment configs
- `--offset`: GPU index offset (default: 0)
- `-j`: Number of concurrent jobs per GPU (default: 1)


### Feature Extraction

For transfer learning experiments

```bash
python feature_extractor.py
```

This extracts ResNet-18 features from CIFAR-10/100 and saves them for use with the `cifar10_feature` and `cifar100_feature` dataset configurations.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{koloskova2025certified,
  title={Certified Unlearning for Neural Networks},
  author={Koloskova, Anastasia and Allouah, Youssef and Jha, Animesh and Guerraoui, Rachid and Koyejo, Sanmi},
  journal={arXiv preprint arXiv:2506.06985},
  year={2025}
}
```
