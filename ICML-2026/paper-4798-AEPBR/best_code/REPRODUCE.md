# Reproducing the Experiments

Run commands from the repository root unless noted otherwise.

## Environment

Create one shared environment for all experiments:

```bash
conda env create -f environment.yml
conda activate approximate-equivariance-projection
```

Then install the Section 4.1 package in editable mode:

```bash
python3 -m pip install -e ./experiments/section_4_1_mlp
```

## Slurm Submission

All `scripts/reproduce_4_*.sh` wrappers can submit their run commands through a Slurm submit file:

```bash
bash scripts/reproduce_4_4.sh --slurm-submit slurm_scripts/... 
```

The submit file is passed to `sbatch` before the actual command, equivalent to:

```bash
sbatch slurm_scripts/... python ...
```

## Section 4.1: MLP Toy Experiments

Generate the main Section 4.1 neural network paper figures:

```bash
bash scripts/reproduce_4_1.sh
```

This writes:

- `experiments/section_4_1_mlp/figures/nn_approx_invariance_linear.{png,pdf}`
- `experiments/section_4_1_mlp/figures/nn_wavey_rings_lambda_grid_one_row.{png,pdf}`
- `experiments/section_4_1_mlp/figures/nn_approx_invariance_lambda_grid.{png,pdf}`
- `experiments/section_4_1_mlp/figures/nn_wavey_rings_lambda_grid.{png,pdf}`

To generate a subset, specify the figure or plot. For example, run:

```bash
bash scripts/reproduce_4_1.sh figure4 figure7
```


## Section 4.2: Approximately Equivariant Dynamics Models

This repository reproduces the PhiFlow smoke plume experiments for Section 4.2. The canonical entrypoint is:

```bash
bash scripts/reproduce_4_2.sh
```

By default this runs the tuned projection-regularized grid: the valid group-convolution and steerable cases over five random seeds. This is 25 runs: Translation group, Rotation steerable, Scale steerable, Rotation group, and Scale group.

Useful subsets and smoke checks:

```bash
bash scripts/reproduce_4_2.sh --dry-run
bash scripts/reproduce_4_2.sh --num-epoch 30 --seeds "0" steerable
bash scripts/reproduce_4_2.sh group
bash scripts/reproduce_4_2.sh translation group
bash scripts/reproduce_4_2.sh --symmetry rotation --model steerable
bash scripts/reproduce_4_2.sh scale
```

The optional symmetry selector is one of `translation`, `rotation`, `scale`, or `all`. The optional model selector is one of `group`, `steerable`, or `all`. All symmetry/model combinations are accepted except `translation steerable`, because the Section 4.2 translation model has no steerable variant. Section 4.2 runs use `--regulariser=projection` only.

The script runs `experiments/section_4_2_dynamics/run_model_reproduce_4_2.py`, which receives all tuned hyperparameters on the command line. W&B logging is disabled by default. With the default settings, metrics are printed to stdout/stderr; under Slurm they will appear in the Slurm output/error logs configured by the submit script.

To enable W&B logging, pass a W&B project to the wrapper:

```bash
export WANDB_API_KEY=...                # or run `wandb login` once
bash scripts/reproduce_4_2.sh --wandb-project approx_equiv_smoke
```

If `--wandb-project` is omitted or empty, `scripts/reproduce_4_2.sh` does not pass W&B arguments and the Python runner does not initialize W&B.

### PhiFlow Smoke Plume Data

The runner expects preprocessed PhiFlow tensors at exactly these paths:

```text
experiments/section_4_2_dynamics/PhiFlow/Translation/raw_data_*.pt
experiments/section_4_2_dynamics/PhiFlow/Rotation/raw_data_*.pt
experiments/section_4_2_dynamics/PhiFlow/Scale/raw_data_*.pt
```

There are two ways to obtain them.

Option 1: download the preprocessed data.

The original authors provide preprocessed smoke plume and JetFlow datasets here:

```text
https://roselab1.ucsd.edu/seafile/d/8886a9ee4c5248afab26/
```

Download the smoke plume/PhiFlow data and place or extract the three folders so the final layout is:

```text
experiments/section_4_2_dynamics/PhiFlow/Translation/
experiments/section_4_2_dynamics/PhiFlow/Rotation/
experiments/section_4_2_dynamics/PhiFlow/Scale/
```

Option 2: generate the PhiFlow data locally.

The original upstream instructions are:

```bash
git clone -b 2.0.1 --single-branch https://github.com/tum-pbs/PhiFlow.git
```

Then copy `experiments/section_4_2_dynamics/data_gen.ipynb` inside that PhiFlow folder and run the notebook. The notebook generates the Translation, Rotation, and Scale smoke plume datasets in folders named `Translation/`, `Rotation/`, and `Scale/` relative to the notebook working directory. 

After generation, the three generated folders must remain under:

```text
experiments/section_4_2_dynamics/PhiFlow/
```


## Section 4.4: Partial escnn

Run the optimal-hyperparameter seed sweep for Table 3:

```bash
bash scripts/reproduce_4_4.sh
```

This defaults to the `table3` target. The same target can be requested explicitly with:

```bash
bash scripts/reproduce_4_4.sh table3
```

To train the 3D CNN, SO3 SCNN, partial CNN RPP, and PenalizedApproxSO3 models for Table 4 and then benchmark their final checkpoints, run:

```bash
bash scripts/reproduce_4_4.sh table4
```

W&B logging is disabled by default for Section 4.4 as well. With the default settings, metrics are printed to stdout/stderr; under Slurm they will appear in the Slurm output/error logs configured by the submit script.

To enable W&B logging, pass a project to the wrapper:

```bash
export WANDB_API_KEY=...                # or run `wandb login` once
bash scripts/reproduce_4_4.sh --wandb-project penalised_run table3
```

If your W&B account uses an entity, pass it explicitly:

```bash
bash scripts/reproduce_4_4.sh --wandb-project penalised_run --wandb-entity your-wandb-entity table4
```

The default W&B mode is `online` when `--wandb-project` is set. To create local W&B runs without syncing, use:

```bash
bash scripts/reproduce_4_4.sh --wandb-project penalised_run --wandb-mode offline table3
```

If `--wandb-project` is omitted or empty, `scripts/reproduce_4_4.sh` does not pass W&B arguments and `medical_mnist2d.py` uses only regular stdout/stderr logging.