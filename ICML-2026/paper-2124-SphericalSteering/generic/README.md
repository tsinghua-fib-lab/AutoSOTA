<div align="center">

### Evaluation Pipeline for Multiple-Choice Reasoning Benchmarks beyond TruthfulQA

</div>

---

## Data

Supported datasets:
- `copa`
- `storycloze`
- `mmlu`
- `winogrande`
- `boolq`

Artifacts are written to:
- `features_generic/`
- `prototypes_generic/`

## Reproduce

### Quick Reproduce (Provided Config Set)

Run the predefined sweep in `quick_set_generic.sh`:

```bash
bash quick_set_generic.sh
```

This script runs 5 datasets with fixed layer/alpha/beta settings:
- BoolQ
- COPA
- StoryCloze
- MMLU
- Winogrande

### Manual Reproduce

Step 1: extract features

```bash
python get_activations_generic.py \
  --model_name llama3.1-8B-Instruct \
  --dataset storycloze \
  --split train \
  --layer 27
```

Step 2: compute prototypes

```bash
python get_prototypes_generic.py \
  --feature_file ./features_generic/llama3.1-8B-Instruct_storycloze_train_l27.npz
```

Step 3: evaluate steering

```bash
python evaluate_generic.py \
  --model_name llama3.1-8B-Instruct \
  --dataset storycloze \
  --layer 27 \
  --prototype_path ./prototypes_generic/llama3.1-8B-Instruct_storycloze_train_l27_proto.npz \
  --kappa 20 --alpha 0.9 --beta -0.7
```

Baseline (no steering):

```bash
python evaluate_generic.py \
  --model_name llama3.1-8B-Instruct \
  --dataset mmlu_global \
  --layer 27 \
  --prototype_path ./prototypes_generic/llama3.1-8B-Instruct_storycloze_train_l27_proto.npz \
  --disable_steering
```

## Key Modules

| Module | Description |
|---|---|
| `get_activations_generic.py` | Extract last-token hidden activations for training pairs |
| `get_prototypes_generic.py` | Compute contrastive prototypes from extracted features |
| `evaluate_generic.py` | Evaluate steering on benchmark tasks with MC1-style scoring |
| `utils_generic.py` | Dataset loaders, split rules, prompt formatting, feature extraction helpers |
| `quick_set_generic.sh` | Predefined multi-dataset run script with tuned hyperparameters |
