# TEDBench Baselines

Standalone training and evaluation scripts for `gotennet`, `bige3nn`, and
`mace`.

## Install

```bash
pip install -r requirements.txt
pip install git+https://github.com/ACEsuit/mace-layer.git
```

`gotennet[full]` provides `GotenNetWrapper`. The MACE baseline imports
`mace_layer` from [ACEsuit/mace-layer](https://github.com/ACEsuit/mace-layer).

## Graph Caches

The scripts train and evaluate on PyTorch Geometric graph caches saved as `.pt`
files. If `train_graphs.pt`, `val_graphs.pt`, or `test_graphs.pt` are missing,
the scripts build them from the TED dataset and save them for reuse.

Put the TED dataset at the path passed with `--dataset_root`. The default is:

```text
datasets/afdb_FS_plddt80
```

Expected layout:

```text
datasets/afdb_FS_plddt80/
  raw/
  processed_files/
  ted_365m.domain_summary.cath.globularity.taxid.tsv.gz
```

On the first run, if graph cache files are missing, the script loads the raw
dataset, preprocesses the protein structures into PyTorch Geometric graphs, and
saves them as `.pt` files:

```text
train_graphs.pt
val_graphs.pt
test_graphs.pt
```

External datasets are passed with `--external_dataset_root`. For example:

```text
datasets/CATH4.4
```

## Train

```bash
python train.py --model gotennet
python train.py --model bige3nn
python train.py --model mace
```

Train with explicit cache and checkpoint paths:

```bash
python train.py \
  --model gotennet \
  --train_graphs_path train_graphs.pt \
  --val_graphs_path val_graphs.pt \
  --test_graphs_path test_graphs.pt \
  --checkpoint_dir checkpoints
```

Enable W&B logging by passing a project name:

```bash
python train.py --model gotennet --wandb_project <project>
```

By default, training reads `train_graphs.pt` and `val_graphs.pt` and saves the
best checkpoint to:

```text
checkpoints/best_model_<model>_seed<seed>.pt
```

Use `--skip_final_test` to skip the final test-set evaluation after training.

## Evaluate

Evaluate one checkpoint on the TED test set:

```bash
python test_model.py \
  --model gotennet \
  --model_path checkpoints/best_model_gotennet_seed42.pt \
  --test_graphs_path test_graphs.pt
```

Evaluate one checkpoint on an external dataset:

```bash
python test_model.py \
  --model gotennet \
  --model_path checkpoints/best_model_gotennet_seed42.pt \
  --external_dataset_root datasets/CATH4.4 \
  --external_dataset_name cath4.4 \
  --external_split test
```

Evaluate all three baselines on the same test graph cache:

```bash
python eval_all.py \
  --checkpoint_dir checkpoints \
  --seed 42 \
  --test_graphs_path test_graphs.pt
```

Evaluate multiple seeds and report mean/std:

```bash
python eval_all.py \
  --checkpoint_dir checkpoints \
  --seeds 42 50 64 \
  --test_graphs_path test_graphs.pt
```

Evaluate all three baselines on an external dataset:

```bash
python eval_all.py \
  --checkpoint_dir checkpoints \
  --seeds 42 50 \
  --external_dataset_root datasets/CATH4.4 \
  --external_dataset_name cath4.4 \
  --external_split test
```

Use `--gotennet_checkpoint`, `--bige3nn_checkpoint`, or `--mace_checkpoint` to
override individual checkpoint paths.
