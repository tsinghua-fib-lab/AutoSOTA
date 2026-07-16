# Baseline (Homoglyph) — Modified Evaluation Code

This folder contains **modified code for baseline evaluation**, designed to reproduce the **baseline paper’s evaluation protocol** as faithfully as possible. It provides:

- **Baseline-style dataset generation** (homoglyph perturbation baseline)
- **Baseline evaluation** (loss-based / unicode_properties-style scoring and outputs)
- **Training configuration consistent with the main experiments**
- `result-analysis` folder — visualization scripts **(Experiment 4.6, Figure 5 in the paper)** for baseline experiments
  
The **usage pattern is identical to the main experiment code** under `src/` (same CLI style, same joblist logic, same directory conventions).

---

## Important dependency note (required for exact baseline compatibility)

To match the baseline paper’s measurement method **exactly**, you must copy the following from the **original baseline repository** into **this directory** before running:

- the `unstealthy/` folder
- `score.py`
- `utils.py`

This is required because the baseline scoring logic and utilities are implemented there, and we keep them unmodified to preserve fidelity.

> **Action required before running**
>
> 1. Go to the original baseline repo  
> 2. Copy `unstealthy/`, `score.py`, and `utils.py`  
> 3. Paste them into this folder (same level as the modified baseline scripts)

---

## What’s included

---

## 1) Baseline dataset generation — `watermark.py`

### What it does
This script generates **homoglyph-based watermark datasets** using the official
`unicode_properties` perturbation logic (via `src.data.perturb_modified`), with
additional controls for scalability, and computational efficiency.

Specifically, it:
- Loads one of the supported raw text sources: `blog1k`, `poems`, or `cnn_news`.
- Selects `--train-size-total` eligible documents (minimum length enforced).
- Assigns documents deterministically to users `user_0001`, `user_0002`, …, `user_<N>`.
- Applies **homoglyph perturbations** to selected documents using the official
  `perturb_dataset` routine.
- Builds a final `train.jsonl` containing both watermarked and non-watermarked texts.

### Advanced user-dependent logic
To remain fully consistent with the official `unicode_properties` experimental
protocol while significantly reducing GPU and storage costs, users are handled
asymmetrically:

- **All users**
  - Receive homoglyph-perturbed texts that are included in the training set.

- **Users `user_0001` to `user_0005` only**
  - Additionally generate:
    - a **null distribution**, and
    - `propagation_inputs.csv` files in the official evaluation format.

- **Users `user_0006+`**
  - Do **not** generate null samples,
  - Do **not** generate propagation inputs,
  - But their homoglyph-perturbed texts are still included in `train.jsonl`.

The null distribution is used only as a representative statistic; limiting it to
the first five users preserves statistical validity while greatly reducing
computational overhead.

### Outputs
For a given `--output-dir-root <ROOT>` and `--version <VERSION>`, the script writes:

```
<ROOT>/<VERSION>/
  ├── train.jsonl
  ├── users/
  │   └── user_0001_watermarks.jsonl
  │   └── user_0002_watermarks.jsonl
  │   └── ...
  └── propagation_inputs/
      └── user_0001_k<K>_null<N>_propagation_inputs.csv
      └── ...
```

### CLI arguments (exact)

Core:
- `--source {blog1k,poems,cnn_news}` (default: `blog1k`)
- `--train-size-total INT` (default: `1000`)
- `--num-users INT` (default: `50`)
- `--wm-per-user INT` (default: `40`)
- `--seed INT` (default: `42`)
- `--version STR` (default: `xp_homoglyph_advanced`)
- `--output-dir-root PATH` (default: `./`)

Dataset inputs:
- `--blog1k-csv PATH`
- `--poems-csv PATH`
- `--cnn-news-csv PATH`

Homoglyph / baseline controls:
- `--base-seed INT` (default: `123`)
- `--group-folder {sampled_perturbation,constant_perturbation}`
- `--null-n-seq INT` (default: `100`)

External dependencies:
- `--external-lib-root PATH`
  (alternative: set `EXTERNAL_LIB_ROOT` in the environment)

---

## 2) Training consistent with main experiments
Model training uses the **same training setup as the main experiments**, including:
- the same LoRA setup and hyperparameters
- the same reproducibility settings (seed control)
- the same FA2 / attention implementation flags (when enabled)

---


## 3) Baseline evaluation — `score_unicode_properties.py`

### What it does
This script implements **unicode_properties-style loss-based scoring** using a
base language model together with a **LoRA adapter**, without requiring a merged
`final_model` on disk.

Key characteristics:
- Loads the **base model + LoRA adapter** in-memory using PEFT.
- Reads `propagation_inputs.csv` files generated during dataset construction.
- Computes **per-token negative log-likelihood losses** using the *same*
  single-text loss function as the original baseline (`_calculate_loss_str`
  imported from `score.py`).
- Adds **batch-level orchestration** to reduce Python overhead while preserving
  the exact per-document loss semantics.

This design ensures strict fidelity to the baseline paper’s scoring logic while
improving practical throughput.

### Input format
- `propagation_inputs.csv`
  - Column: `text`
  - **Row 0**: scalar `k` (number of evaluated documents)
  - **Rows 1..k**: input documents to be scored

This format is identical to the official baseline pipeline.

### Outputs
- A CSV file containing per-token losses:
  - **Row 0**: `[k]`
  - **Row 1..k**: token-level loss values for each document

The output format is drop-in compatible with downstream baseline analysis and
aggregation scripts.

### Model loading behavior
- The tokenizer is loaded from `--base_model_id`.
- If the tokenizer has no pad token, `eos_token` is used.
- Long documents are **left-truncated**, keeping the last
  `--unicode_max_length` tokens, matching baseline behavior.
- The LoRA adapter is loaded via PEFT and **not merged** into the base model.

### CLI arguments (exact)

Required:
- `--base_model_id STR`  
  Hugging Face model ID or local path to the base model.
- `--adapter_path PATH`  
  Path to the LoRA adapter directory.
- `--path_to_inputs PATH`  
  Path to `propagation_inputs.csv`.
- `--output_score_path PATH`  
  Output CSV path for per-token losses.

Optional:
- `--unicode_max_length INT` (default: `4096`)  
  Number of last tokens used for scoring.
- `--attn_impl {eager,sdpa,flash_attention_2}` (default: `flash_attention_2`)  
  Attention backend (`eager` is the most stable).
- `--batch_size INT` (default: `32`)  
  Number of documents processed per outer loop batch.

---
## How to run

### Run style
The workflow is intentionally kept **identical** to the main experiment code in `src/`:
- same `--data_base`, `--wm_version`, `--model_id`, etc.
- same output directory conventions

So if you already know how to run the main experiments under `src/`, you can run the baseline version the same way—just point to the baseline scripts / baseline dataset versions.

### Typical pipeline
1. **Generate baseline dataset** (homoglyph baseline + limited null/propagation for selected users)
2. **Train model** with the same settings as main experiments
3. **Evaluate** with baseline-compatible unicode_properties/loss scoring

All scripts follow the same CLI conventions as the main pipeline.

---

## Outputs

The pipeline writes outputs using the same high-level structure as the main experiments, including:
- `train.jsonl` for training
- per-user manifests in `users/`
- `propagation_inputs/` (for the selected users)
- evaluation results in `probe_outputs_unicode_properties/<wm_version>/`
- archived experiment bundles under `experiments/...`
- a `metadata.json` describing the run configuration

---

## Notes

- This code is **only** for baseline evaluation (homoglyph baseline) and is not intended to replace the main experiment pipeline.
- The goal is strict comparability with the baseline paper, while keeping training and orchestration aligned with the main experiment codebase.
---

## Command-line usage (quick reference)

### 1) Dataset generation (homoglyph baseline)

```bash
python3 watermark.py \
  --source blog1k \
  --train-size-total 1000 \
  --num-users 50 \
  --wm-per-user 40 \
  --seed 42 \
  --base-seed 123 \
  --null-n-seq 100 \
  --group-folder sampled_perturbation \
  --version xp_homoglyph_advanced \
  --output-dir-root ./data \
  --external-lib-root /path/to/datawatermarks
```

---

### 2) Baseline evaluation (unicode_properties loss scoring)

```bash
python3 score_unicode_properties.py \
  --base_model_id ./models/base_model \
  --adapter_path ./data/outputs/model_A/lora-single-XP1-T1000_U1_P100_seed123-seed42/final_adapter \
  --path_to_inputs ./data/XP1-T1000_U1_P100_seed123/propagation_inputs/user_0001_k100_null100_propagation_inputs.csv \
  --output_score_path ./data/XP1-T1000_U1_P100_seed123/scores/user_0001_scores.csv \
  --unicode_max_length 4096 \
  --attn_impl flash_attention_2 \
  --batch_size 2
```
