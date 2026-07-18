# SOTA Preparation Repair — Paper 6256

## Original Failure

The normal SOTA preparation path failed for paper 6256 with two errors:

1. **Docker overlay full**: The Docker storage pool at `/docker_data` was exhausted (200G/200G, 0 available). This prevented:
   - `apt-get update` (E: List directory missing — No space left on device)
   - `git init` (git not installed)
   - Creating a new container from `autosota/paper-6256:env`
   - Writing any files inside the container

2. **Docker authorization plugin**: Host networking and proxy options were rejected by `ehub.ctcdn.cn/bc-ops/opa-docker-authz-v2:0.1`.

## Repair Steps

1. **Freed Docker storage**: Removed 18 dangling `<none>:<none>` images via `docker rmi`, recovering ~17GB of space in the overlay pool.
2. **Used existing container**: `autosota_repro_paper_6256` was still running (from the reproduction phase). Instead of creating a new container, we reused it.
3. **Installed git**: `apt-get install -y git` (now works with free space).
4. **Initialized git repo**: `git init`, baseline commit, `_baseline` tag.
5. **Copied record_score.sh**: From host to `/tools/record_score.sh` in the container.
6. **Created scores.jsonl**: At `/autosota_artifacts/paper-6256/sota/scores.jsonl`.

## Corrected Evaluation Command

The manifest eval_command is correct and runs successfully inside the container:

```bash
unset HF_ENDPOINT && \
export HF_HOME=/autosota_cache/hf && \
export HUGGINGFACE_HUB_CACHE=/autosota_cache/hf && \
export TRANSFORMERS_CACHE=/autosota_cache/hf && \
export TORCH_HOME=/models/torch && \
export RECYCLING4VLALIGNMENT_WEIGHTS_DIR=/models/paper-6256/weights && \
export RECYCLING4VLALIGNMENT_DATA_DIR=/datasets/paper-6256/data && \
export RECYCLING4VLALIGNMENT_CHECKPOINT_DIR=/autosota_cache/checkpoints_fewshot && \
export RECYCLING4VLALIGNMENT_EMBEDDINGS_DIR=/models/paper-6256/embeddings && \
export TMPDIR=/autosota_cache/tmp && \
cd /repo && \
python3 classification_and_retrieval.py \
  --image_models "timm/beit_base_patch16_224.in22k_ft_in22k" \
  --text_models "clip_vitb32" \
  --task classification \
  --mode MLP \
  --datasets cifar100 \
  --dataset_img_repr cifar100 \
  --few_shot_samples 4 \
  --sequential_training \
  --epochs 200 \
  --seed 9871
```

## Baseline Evidence

- **Measured**: Top-1 Accuracy: 76.39%, Top-5 Accuracy: 93.83%
- **Manifest baseline**: 76.64% (seed 9871)
- **Delta**: -0.25pp — within normal numerical noise (CUDA non-determinism, slight environment differences)
- **Recorded**: Iteration 0 in scores.jsonl, commit `707b049f`

## Reusable Resources

| Path | Contents |
|------|----------|
| `/models/paper-6256/weights/` | Pre-downloaded BEiT weights (timm_beit_base_patch16_224.in22k_ft_in22k) |
| `/datasets/paper-6256/data/` | CIFAR-100 dataset (pre-extracted) |
| `/models/paper-6256/embeddings/` | (empty, for generated embeddings) |
| `/autosota_cache/checkpoints_fewshot/` | MLP alignment checkpoints |
| `/autosota_cache/hf/` | HuggingFace cache (2GB, includes BEiT hf model) |

## Safe Optimization Targets

The evaluation uses:
- **Image encoder**: BEiT-B/16 (timm/beit_base_patch16_224.in22k_ft_in22k) — 768-dim features
- **Text encoder**: CLIP ViT-B/32 — 512-dim text embeddings
- **Alignment**: MLP (512→3072→768) with GELU + LayerNorm + Dropout(0.5)
- **Training**: Sequential training: Stage 1 (500 epochs with Adam, lr=5e-3, CosineEmbeddingLoss), Stage 2 (200 epochs with Adam, lr=1e-3)
- **Evaluation**: Cosine similarity between aligned text prototypes and image features

Safe change areas (from manifest `known_levers` and code analysis):
1. Learning rate (5e-3 for stage 1, 1e-3 for stage 2)
2. Architecture (two_layer vs single layer, hidden dimensions)
3. Dropout (0.5 MLP, 0.3 input)
4. Optimizer choice (Adam default)
5. Epochs (Stage 1: 500, Stage 2: 200)
6. Batch size (512)
7. Prompt template for text encoding
8. N-shot (4)
9. Weight preprocessing method (mean/attention/linear)
10. Loss function (cosine embedding)
11. Cosine annealing LR schedule

## Key Files

| File | Purpose |
|------|---------|
| `classification_and_retrieval.py` | Main entry point |
| `alignment/mlp_img_aligner/train_aligners.py` | Training logic |
| `alignment/mlp_img_aligner/aligned_models.py` | Model definitions (TextToImageMLP, ClassificationModel) |
| `utils/utils.py` | Backbone loading, get_backbone() |
| `dataloaders/` | Dataset loading |
