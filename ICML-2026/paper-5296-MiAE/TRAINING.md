# Training Guide

Commands and hyperparameters for reproducing all TEDBench results from the paper.

All scripts use [Hydra](https://hydra.cc) for configuration.  Override any config
key on the command line with `key=value`; add new keys with `+key=value`.

---

## 1. Pretraining MiAE

Pretrain on the Foldseek-clustered AFDB with the default MiAE-B (102 M) model
and mask ratio 0.9.

```bash
# Single GPU (local data)
python main_pretrain.py

# Single GPU (from HuggingFace)
python main_pretrain.py datamodule=hf_afdbfs

# Multi-GPU with SLURM (4 nodes × 8 H200 GPUs; effective batch size 4096)
srun python main_pretrain.py experiment=tedbench_base_n4g8

# MiAE-L (339 M)
srun python main_pretrain.py \
    experiment=tedbench_base_n4g8 \
    model.name=miae_l \
    datamodule.batch_size=16 \
    trainer.accumulate_grad_batches=8
```

| Hyperparameter | Value |
|---|---|
| Mask ratio | 0.9 |
| Learning rate | 0.0024 |
| Weight decay | 0.05 |
| Effective batch size | 4096 |
| Warmup steps | 5 000 |
| Total steps | 100 000 |

---

## 2. Supervised Training from Scratch

Train a MiAEEncoder directly on TEDBench without pretraining.

```bash
# MiAE-B (default)
python main_finetune_ted.py

# With HuggingFace dataset
python main_finetune_ted.py datamodule=hf_ted

# Multi-GPU with SLURM (8 H200 GPUs; effective batch size 4096)
srun python main_finetune_ted.py experiment=scratch_ted_base_n1g8

# MiAE-B+seq
python main_finetune_ted.py model.use_seq_input=true
```

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.0016 |
| Weight decay | 0.1 |
| Effective batch size | 4096 |
| Warmup steps | 1 830 |
| Total steps | 18 300 (~200 epochs) |
| Coordinate noise std | 0.2 Å |

---

## 3. Fine-tuning from a Pretrained Checkpoint

Initialize from a pretrained MiAE and fine-tune end-to-end with layer-wise LR decay.

```bash
# MiAE-B from HuggingFace Hub
python main_finetune_ted.py \
    pretrained_model_path=TEDBench/miae-b \
    experiment=finetune_ted_base_n1g8

# MiAE-B+seq
python main_finetune_ted.py \
    pretrained_model_path=TEDBench/miae-b-seq \
    experiment=finetune_ted_base_n1g8 \
    model.use_seq_input=true

# MiAE-L
python main_finetune_ted.py \
    pretrained_model_path=TEDBench/miae-l \
    experiment=finetune_ted_base_n1g8 \
    model.name=miae_l \
    datamodule.batch_size=32 \
    trainer.accumulate_grad_batches=4

# With HuggingFace dataset
python main_finetune_ted.py \
    datamodule=hf_ted \
    pretrained_model_path=TEDBench/miae-b \
    experiment=finetune_ted_base_n1g8
```

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.0016 |
| Layer-wise LR decay | 0.8 |
| Weight decay | 0.1 |
| Effective batch size | 1024 |
| Warmup steps | 1 830 |
| Total steps | 18 300 (~50 epochs) |

---

## 4. Linear Probing

Extract frozen representations from a pretrained MiAE and fit a linear classifier
with L-BFGS (cross-validated regularisation).

```bash
# MiAE-B from HuggingFace Hub
python main_linprobe_ted.py \
    pretrained_model_path=TEDBench/miae-b

# With HuggingFace dataset
python main_linprobe_ted.py \
    datamodule=hf_ted \
    pretrained_model_path=TEDBench/miae-b

# MiAE-L
python main_linprobe_ted.py \
    pretrained_model_path=TEDBench/miae-l \
    model.name=miae_l
```

Results are saved to `logs/linprobe/<dataset>/<seed>/runs/.../results.csv`.

---

## 5. Testing

Evaluate a trained model on the TEDBench test split and/or the CATH 4.4
external test set.

```bash
# TEDBench test split (fine-tuned MiAE-B)
python main_test_ted.py \
    pretrained_model_path=TEDBench/miae-b-ft

# CATH 4.4 external test set
python main_test_ted.py \
    datamodule=hf_cath_test \
    pretrained_model_path=TEDBench/miae-b-ft

# Supervised-from-scratch MiAE-B
python main_test_ted.py \
    pretrained_model_path=TEDBench/miae-b-sc
```

---

## Reproducing Paper Results

### Supervised from scratch

```bash
# MiAE-B
python main_test_ted.py pretrained_model_path=TEDBench/miae-b-sc

# MiAE-B on CATH 4.4 external test
python main_test_ted.py \
    pretrained_model_path=TEDBench/miae-b-sc \
    datamodule=hf_cath_test

# MiAE-B+seq
python main_test_ted.py \
    pretrained_model_path=TEDBench/miae-b-seq-sc \
    model.use_seq_input=true

# MiAE-L
python main_test_ted.py \
    pretrained_model_path=TEDBench/miae-l-sc \
    model.name=miae_l
```

### Fine-tuned

```bash
# MiAE-B (replace -sc with -ft for the fine-tuned variant)
python main_test_ted.py pretrained_model_path=TEDBench/miae-b-ft

# MiAE-B on CATH 4.4 external test
python main_test_ted.py \
    pretrained_model_path=TEDBench/miae-b-ft \
    datamodule=hf_cath_test
```

### Linear probing

```bash
python main_linprobe_ted.py pretrained_model_path=TEDBench/miae-b
```

---

## Baselines

Baseline scripts are in `baselines/`.  Each script has its own Hydra config under
`baselines/configs/`.

```bash
# ESM2 fine-tuning
python baselines/esm2_finetune_ted.py

# ESM2 testing
python baselines/esm2_test_ted.py train.ckpt_path=<path/to/best_model.ckpt>

# SaProt fine-tuning
python baselines/saprot_finetune_ted.py

# SaProt linear probing
python baselines/saprot_linprobe_ted.py
```

> **Note:** The SaProt baselines require [Foldseek](https://github.com/steineggerlab/foldseek)
> to be installed and `foldseek_path` set in the config.

---

## Config Reference

Key config files:

| File | Purpose |
|---|---|
| `configs/pretrain.yaml` | MiAE pretraining |
| `configs/finetune_ted.yaml` | Fine-tuning / supervised from scratch |
| `configs/linprobe_ted.yaml` | Linear probing |
| `configs/test_ted.yaml` | Evaluation |
| `configs/datamodule/hf_ted.yaml` | TEDBench from HuggingFace |
| `configs/datamodule/hf_cath_test.yaml` | CATH 4.4 from HuggingFace |
| `configs/datamodule/hf_afdbfs.yaml` | AFDB pretraining from HuggingFace |
| `configs/experiment/finetune_ted_base.yaml` | Fine-tuning hyperparameters (Table 6b) |
