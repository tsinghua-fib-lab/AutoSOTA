# Repository Guidelines

## Project Structure & Module Organization
- `benchmarks/` holds dataset wrappers (e.g., BLINK) and evaluation helpers.
- `configs/` contains YAML configs that drive runs (model, cache_dir, dataset, recompute settings).
- `inference/` provides shared inference runners.
- `models/` hosts model adapters and KV-cache logic (`models/qwen/kv_cache/`).
- `scripts/` contains runnable entry points for baselines and recomputation.
- `output/` is used for saved results; notebooks live at the repo root.

## Build, Test, and Development Commands
This repo runs directly with Python scripts (no build step detected).

```bash
# KV cache recomputation run (primary entry point)
python scripts/inference_with_recompute_kv.py --config configs/blink_counting.yaml

# Baseline BLINK inference
python scripts/run_blink.py --config configs/blink_counting.yaml

# Simple demo with visual/text patches
python scripts/qwen3_vlm_inference.py
```

## Coding Style & Naming Conventions
- Python uses 4-space indentation and snake_case for functions/variables.
- Keep module names lowercase (see `models/`, `benchmarks/`).
- Config keys are lowercase with underscores (see `configs/blink_counting.yaml`).
- No formatter or linter is configured; match existing style and avoid reformat-only diffs.

## Testing Guidelines
- No formal test suite is present. Use small-sample smoke tests instead.
- Prefer setting `num_samples` in `configs/blink_counting.yaml` to a small value (e.g., 5) for quick validation.
- Check `output/` artifacts and console metrics (accuracy, timing).

## Commit & Pull Request Guidelines
- Recent commits use bracketed tags like `[INIT]`, `[FEAT]`, `[FIX]`. Follow this pattern (e.g., `[FEAT] Add new scorer`).
- PRs should include a brief summary, the exact config used, and where results were saved (path under `output/`).
- If changes affect outputs or datasets, mention expected metric or behavioral differences.

## Configuration & Data Notes
- Edit `configs/blink_counting.yaml` to change model, cache directory, and recompute settings.
- `cache_dir` should point to a local HF cache; large model files are not stored in-repo.
