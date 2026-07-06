# LTM (Learning Tabular Models)

Unified interface for extracting embeddings and training prediction heads. Supports TP-BERTa, Nomic, and BGE.

## Environment Variables

```bash
export TPBERTA_ROOT="./tp-berta"
export TPBERTA_PRETRAIN_DIR="$TPBERTA_ROOT/checkpoints/tp-joint"
export TPBERTA_BASE_MODEL_DIR="$TPBERTA_ROOT/checkpoints/roberta-base"
export PYTHONPATH="$PROJECT_ROOT:$TPBERTA_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```

## Quick Start

### 1. Generate Embeddings for RelBench (.npy)

```bash
./LTM/scripts/save_embed_numpy.sh
```

**Output Structure**:
```
run_outputs/data/relbench/baselines/ltm/tpberta_relbench/
├── nomic/
│   ├── hm_user-churn_data.npy
│   ├── avito_user-clicks_data.npy
│   └── ...
├── bge/
│   ├── hm_user-churn_data.npy
│   └── ...
└── tpberta/
    ├── hm_user-churn_data.npy
    └── ...
```

**Logs**: `run_outputs/data/relbench/baselines/ltm/logs/run_embeddings_{timestamp}.log`

---

### 2. Preprocess Medium Tables (CSV)

```bash
./LTM/scripts/save_medium_embed_csv.sh              # All
./LTM/scripts/save_medium_embed_csv.sh avito-user-clicks  # Single
```

**Input Structure**:
```
datasets/fit-medium-table/
├── avito-user-clicks/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── target_col.txt
└── ...
```

**Output Structure**:
```
run_outputs/data/relbench/baselines/ltm/tpberta_table/
├── nomic/
│   ├── avito-user-clicks/
│   │   ├── train.csv          # embedding, target
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── feature_names.json
│   └── ...
├── bge/
│   └── ...
└── tpberta/
    └── ...
```

**Datasets**: avito-user-clicks, avito-ad-ctr, event-user-repeat, event-user-attendance, ratebeer-beer-positive, ratebeer-place-positive, ratebeer-user-active, trial-site-success, trial-study-outcome, hm-item-sales, hm-user-churn

---

### 3. Train Prediction Head

```bash
./LTM/scripts/train_ltm.sh            # All
./LTM/scripts/train_ltm.sh avito-user-clicks  # Single
```

**Input Structure**:
```
run_outputs/data/relbench/baselines/ltm/tpberta_table/
├── nomic/
│   └── avito-user-clicks/     # From step 2
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── ...
```

**Output Structure**:
```
run_outputs/data/relbench/baselines/ltm/results/
├── nomic_head/
│   ├── avito-user-clicks/
│   │   ├── results.json       # metrics
│   │   ├── test_predictions.npy
│   │   └── test_targets.npy
│   └── ...
├── bge_head/
│   └── ...
└── tpberta_head/
    └── ...
```

---

## Python API

### Extract Embeddings

```python
from LTM import get_embeddings
import pandas as pd

df = pd.read_csv("data.csv")

# TP-BERTa
emb = get_embeddings(df, model="tpberta", pretrain_dir="...", has_label=False)

# Nomic
emb = get_embeddings(df, model="nomic", task_prefix="classification", batch_size=32)

# BGE
emb = get_embeddings(df, model="bge", batch_size=32)
```


---

## Models

| Model | Type | Config |
|-------|------|--------|
| **TP-BERTa** | Table transformer | Requires `TPBERTA_PRETRAIN_DIR` |
| **Nomic** | Text embedding | Task prefix: `"classification"`, `"search_document"`, etc. |
| **BGE** | Text embedding | No special config |

---
