# cpc_llm

Conformal Policy Control for LLMs. An iterative framework for training language models with formal risk-control guarantees on the generated outputs.

## How it works

The system iteratively improves a language model policy while maintaining statistical guarantees:

1. **Generate** sequences from the current policy at multiple temperatures
2. **Score** sequences using a test function (e.g., Ehrlich protein motif discovery)
3. **Compute likelihoods** of sequences under all previous model checkpoints
4. **Run CPC** to find the most aggressive policy bound (beta) satisfying the risk level (alpha)
5. **Sample** from the CPC-constrained policy via acceptance-rejection
6. **Train** a new policy (SFT, DPO, or MARGE) on the constrained outputs
7. Repeat — each iteration adds calibration data for the next round of CPC

## Entry point

`main.py` — Hydra-decorated entry point. Config at `config/cpc_llm.yaml` (with `smoke.yaml` for quick tests). Run via:
```
cpc-llm --config-name=cpc_llm
```

## Module overview

- `calibrate/` — CPC algorithm: beta search, likelihood constraining, normalization
- `core/` — Model inference and likelihood computation (ModelClient, HuggingFace wrappers)
- `data/` — Dataset generation, formatting (edit pairs, preference pairs), splitting
- `infer/` — Sequence generation and acceptance-rejection sampling
- `infrastructure/` — File handling (local/S3), pipeline orchestration, SLURM
- `train/` — Policy training (SFT, DPO, MARGE)
- `test_functions/` — Ehrlich test function utilities and metrics

## Key config parameters

- `conformal_policy_control.alpha` — risk level (e.g., 0.1)
- `num_sft_rounds`, `num_dpo_rounds`, `num_marge_rounds` — training iterations per method
- `proportion_of_old_data` — fraction of historical data to mix with current round
- `proportion_of_old_seeds` — fraction of seeds from previous iterations
- `parent_output_dir` — S3 or local path for outputs

## Dependencies

Declared in `pyproject.toml`. Key packages: torch, transformers, trl, hydra-core, botorch, pytorch-holo, wandb.
