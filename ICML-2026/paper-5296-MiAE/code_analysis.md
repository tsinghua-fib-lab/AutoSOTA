# SOTA Preparation Repair — Paper 5296

## Original Failure

The normal SOTA preparation path failed because:
1. The reusable reproduction container `autosota_repro_paper_5296` did not have `git` installed, and `apt-get` couldn't reach Ubuntu repositories due to proxy configuration.
2. A fresh container `autosota_sota_paper_5296` was started from `autosota/paper-5296:reproduced`, but the same issue persisted.
3. **Fix**: Unset HTTP_PROXY/HTTPS_PROXY environment variables before running `apt-get`, allowing direct access to Ubuntu repositories. Then `apt-get install -y git` succeeded.

## Repaired In-Container Evaluation Command

```bash
cd /repo
python main_test_ted.py \
  pretrained_model_path=/models/TEDBench_miae-s-ft \
  datamodule=hf_ted \
  datamodule.root=/datasets/ted \
  model.name=miae_s \
  datamodule.num_workers=0 \
  datamodule.batch_size=16 \
  trainer.devices=1
```

## Baseline Reproduction Evidence

| Metric | Manifest | Reproduced | Status |
|--------|----------|------------|--------|
| Accuracy | 79.16 | 79.16 | ✓ Exact match |
| Macro F1 | 72.28 | 72.28 | ✓ Exact match |

Both metrics reproduced exactly using the fine-tuned MiAE-S checkpoint at `/models/TEDBench_miae-s-ft` and TEDBench dataset at `/datasets/ted`.

## Reusable /paper_data Resources

Mounted at `/paper_data` (read-only):
- **Model checkpoints**: MiAE-S/B/L (pretrained, fine-tuned, supervised-from-scratch, sequence variants)
- **Datasets**: TEDBench (TED, AFDB, CATH), ESM2-650M, SaProt-650M, ESM3-small
- **Pre-downloaded archives**: MPCDF_CATH44_archive.tar.gz, MPCDF_TED_AFDB_archive.tar.gz

All resources already copied/configured during reproduction. No additional downloads needed for fine-tuning experiments.

## Safe Optimization Targets

- **Loss function**: `tedbench/model/pl_engine.py:176-177` (CrossEntropyLoss → FocalLoss, label smoothing)
- **Training step**: `tedbench/model/pl_engine.py:237-238` (logit adjustment, mixup, SupCon)
- **DataLoader**: `tedbench/data/datasets.py:121-128` (balanced sampling)
- **Test step**: `tedbench/model/pl_engine.py:258-260` (TTA)
- **Hyperparameters**: `configs/finetune_ted.yaml` (LR, weight decay, warmup, layer-wise LR decay)
- **Model head**: `tedbench/model/fot.py:71` (linear → cosine classifier)
- **Transforms**: `tedbench/data/transform.py:22-65` (noise, crop)
