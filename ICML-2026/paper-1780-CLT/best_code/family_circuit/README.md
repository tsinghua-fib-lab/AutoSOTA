# Family Circuit

This folder contains code for protein family-based circuit discovery, which identifies circuits by training linear probes on InterPro family classifications and using attribution methods.

## Key Components

- `01_extract_embeddings.py`: Extracts and caches ESM2 embeddings for protein families from InterPro annotations
- `02_discover_circuits_clt.py`: Circuit discovery pipeline for CLT (Cross-Layer Transcoder) models
- `02_discover_circuits_plt.py`: Circuit discovery pipeline for PLT (Per-Layer Transcoder) models
- `family_utils.py`: Utilities for data loading, probe training, and circuit evaluation
- `main.sh`: Shell script for running the complete family circuit discovery pipeline
- `families/`: Output directory containing discovered circuits and cached embeddings

## Usage

Run the complete family circuit discovery pipeline:

```bash
./main.sh
```

Run with specific options:

```bash
# Limit to top 10 families
./main.sh --limit 10

# Target specific InterPro family
./main.sh --target IPR000724

# Overwrite existing results
./main.sh --overwrite
```

## Data

Uses InterPro family annotations to create positive/negative classification datasets for each family, with balanced sampling and train/validation/test splits. You can find the family data for ESM2-8M
at https://huggingface.co/datasets/ktalreja/ProtoMechData/blob/main/families.tar.gz and family data for ESM2-35M at https://huggingface.co/datasets/ktalreja/ProtoMechData/blob/main/families_35M.tar.gz
(unzip first).