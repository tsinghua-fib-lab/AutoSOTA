# Steering

This folder contains code for probe and DMS (Deep Mutational Scanning) steering experiments, which test the causal effects of discovered protein circuits by ablating or modifying latents.

## Key Components

- `run_probe_steering.py`: Main orchestration script for probe steering experiments
- `steering_utils.py`: Core utilities for latent ablation and circuit steering
- `full_replacement_models.py`: Model classes for full circuit replacement (CLT/PLT)
- `local_replacement_models.py`: Model classes for local replacement
- `find_steering_circuit_attribution_sampler.py`: Circuit discovery for steering
- `scoring_utils.py`: Evaluation and scoring functions
- `gen_utils.py`: General utility functions
- `main_probe_steering.sh`: Shell script for running probe steering experiments
- `eval_models/`: Pre-trained CNN models for held-out evaluation

## Usage

Run probe steering experiments:

```bash
./main_probe_steering.sh
```

Or run individual experiments:

```bash
python run_probe_steering.py --dms_dir ../function_circuit/DMS --clt_ckpt ../models/CLT_L6_D3200/checkpoints/last.ckpt --plt_ckpt ../models/PLT_L6_D3200/checkpoints/last.ckpt --esm_weights ../models/esm2_t6_8M_UR50D.pt --circuit_base ../function_circuit/functions
```

