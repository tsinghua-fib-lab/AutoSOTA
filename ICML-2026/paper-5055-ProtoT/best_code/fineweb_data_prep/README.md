# FineWeb NPZ Splits

Code to create the FineWeb-Edu NPZ splits used by `prototype_attention`.

Outputs:

```text
train.npz
val.npz
test.npz
```

Each NPZ contains:

```text
tokens   flat token id array
offsets  document boundary offsets into tokens
```

## Install

```bash
pip install -r fineweb_data_prep/requirements.txt
```

## Create

Run from the repo root:

```bash
python fineweb_data_prep/prepare_fineweb_npz.py \
  --output-dir data/FineWeb \
  --tokenizer-json tok/fineweb_bpe_16000.json \
  --target-tokens 250000000
```

This writes:

```text
data/FineWeb/train.npz
data/FineWeb/val.npz
data/FineWeb/test.npz
data/FineWeb/dataset_info.json
```

## Verify

```bash
python fineweb_data_prep/verify_fineweb_npz.py \
  --data-dir data/FineWeb \
  --tokenizer-json tok/fineweb_bpe_16000.json
```

## Generated Directory

The generated directory contains:

```text
dataset_info.json
train.npz
val.npz
test.npz
```
