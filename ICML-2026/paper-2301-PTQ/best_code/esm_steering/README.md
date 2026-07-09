# ESM Steering

Code for Contrastive Activation Addition (CAA) steering of ESM-2 model generations.

## Key Components

- `run_caa_steering.py`: Main orchestration script for CAA steering experiments on DMS datasets
- `steer_seq.py`: Core steering logic for individual protein sequences using CAA vectors
- `caa_utils.py`: Utilities for steering
- `main_caa_steering.sh`: Shell script for running CAA steering experiments

## Usage

Run CAA steering experiments:

```bash
./main_caa_steering.sh
```
- Terminal log stored in `caa_steering_log.txt`

Or run individual sequence steering:

```bash
./main_caa_steering.sh [DATASET_NAME]
```
- Terminal log stored in `caa_steering_log_{DATASET_NAME}.txt`

Results stored in `results_caa_steering`

