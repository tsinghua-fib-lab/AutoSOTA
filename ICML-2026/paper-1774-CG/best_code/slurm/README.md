# Slurm scripts

Uniform batch scripts for the cluster. All three experiments are launched the
same way and log to the unified W&B project (`calibrated-guidance`).

## Recommended: `submit.sh` + a local config

Put your private paths (checkpoints, data, partition, python) in a **gitignored**
`env.local.sh` — copy [`env.example.sh`](../env.example.sh) and edit it. Then
launch with [`submit.sh`](../submit.sh), which sources `env.local.sh` and injects
`--partition`:

```bash
cp env.example.sh env.local.sh        # then edit with your real paths
./submit.sh smoke                     # validate all 3 pipelines (1 GPU)
REINF_K=512 POSTERIOR=meanflow ./submit.sh black_hole --array=0-24
NUM_PARTICLES=256 ./submit.sh super_resolution --array=0-19
./submit.sh sbi
```

Experiment knobs are plain environment variables, exported to the job. Nothing
private is ever committed — `env.local.sh` holds the only machine-specific paths.

## Or plain `sbatch`

The scripts also source `env.local.sh` themselves if present, so bare `sbatch`
works too — you just pass `--partition` yourself:

```bash
sbatch --partition=<your-partition> slurm/smoke.sbatch
REINF_K=512 sbatch --partition=<your-partition> slurm/black_hole.sbatch
```

## Knobs (all overridable via env / `env.local.sh`)

| Var | Meaning | Default |
|---|---|---|
| `PY` | python interpreter | `python` |
| `SLURM_PARTITION` | GPU partition (used by `submit.sh`) | `gpu` |
| `BH_PRIOR` / `BH_DATA` | black-hole prior ckpt / data dir | vendored paths |
| `MF_CKPT` / `MF_ROOT` | mean-flow ckpt / easy_meanflow root | — |
| `VAL_ROOT` / `IMAGENET_VAL_TAR` / `SR_CKPT` | super-res data / tar / pMF ckpt | — |
| `REINF_K`, `NUM_STEPS`, `POSTERIOR`, `NUM_PARTICLES`, … | per-experiment knobs | see scripts |
| `CBG_WANDB` / `CBG_WANDB_PROJECT` | W&B toggle / project | on / `calibrated-guidance` |

`black_hole.sbatch` and `super_resolution.sbatch` are **array jobs**: each task
processes a shard, so the 100 black-hole instances / 1000 super-resolution images
run in parallel. Set the array width with `--array=0-N` and the per-task shard
size with `CHUNK`.
