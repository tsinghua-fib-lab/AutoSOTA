# A Large-scale NYC Ride-sharing Simualtor

This repository contains code for running simulations of NYC ride-sharing experiments.

## Simulation Structure

- `ride_sharing.ipynb`: Analysis of Global Average Treatment Effects (GATE) with various estimators
- `rideshare.py`: Main simulation code for ridesharing experiments
- `taxi-zones.parquet`: Taxi zone data for Manhattan
- `manhattan-nodes.parquet`: Node data for Manhattan street network
- `configs/`: Configuration files for different experimental setups (YAML format)
- `output/`: Directory for experiment results

## Setup

1. Install dependencies using [Poetry](https://python-poetry.org/docs/) (recommended for Python 3.11):
```bash
poetry lock
poetry install
```

2. Run experiments using configuration files:
```bash
# For switchback design
poetry run python rideshare.py with configs/switchback.yaml
```

Example configuration file (configs/switchback.yaml):
```yaml
n_cars: 300
k: 100
batch_size: 100
p: 0.5
seed: 42
n_events: 500000
design:
  name: switchback
  switch_every: 600
output: output/switchback/results
config_output: output/switchback/config
```

To reproduce the plots from the paper, the following simulation outputs are required before running the notebook `ride_sharing.ipynb`:

- Simulations under `pure-A` and `pure-B` configurations (used to compute the GATE).
- Simulations under  `switchback` configuration with interval lengths of 600, 1200, 1800, and 3600 seconds.

## Dependencies

See `pyproject.toml` for the complete list of dependencies. Main requirements:
- JAX
- Pandas
- NumPy
- Matplotlib
- Sacred 