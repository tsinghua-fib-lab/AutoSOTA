# Function Circuit

This folder contains code for DMS (Deep Mutational Scanning) function-based circuit discovery, which identifies protein circuits by training CNN probes on fitness scores and using attribution methods.

## Key Components

- `01_discover_circuits.py`: Main pipeline for circuit discovery using CNN probes and CLT/PLT models
- `function_utils.py`: Utilities including CNNProbe class, data loading, and evaluation functions
- `main.sh`: Shell script for running the complete circuit discovery pipeline
- `DMS/`: DMS data with cross-validation folds (single and multiple substitutions)
- `functions/`: Discovered circuits for different methods (CLT_direct, CLT_sequential, PLT, etc.)
- `probe/`: Trained CNN probes for function prediction
- `embeddings_cache/`: Cached ESM2 embeddings for efficiency

To run the experiments from the paper, create the DMS folder and add a CSV of the DMS data. Multiple substitutions should go into `DMS/cv_folds_multiples_substitutions/` (available [here](https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_multiples_substitutions.zip)). and single substitutions `DMS/cv_folds_singles_substitutions/` (available [here](https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip)). 

You can find the probes for ESM2-8M in `probe/`on the HuggingFace repository [here](https://huggingface.co/ktalreja/ProtoMechModels/tree/main/probe) and probes for ESM2-35M in `probe_35M/`on the HuggingFace repository [here](https://huggingface.co/ktalreja/ProtoMechModels/tree/main/probe_35M).

## Usage

Run the complete circuit discovery pipeline:

```bash
./main.sh
```

## Methods

- **CLT_direct**: Direct cross-layer transcoder circuits
- **CLT_sequential**: Sequential CLT circuits with frozen attention
- **CLT_sequential_no_frozen**: Sequential CLT circuits without frozen attention
- **PLT**: Per-layer transcoder circuits
- **PLT_no_frozen**: PLT circuits without frozen attention

## Output

Generates circuit JSON files with node attributions, Spearman correlations, and NMSE scores for each DMS dataset and method combination. You can find the function data for ESM2-8M
at https://huggingface.co/datasets/ktalreja/ProtoMechData/blob/main/functions.tar.gz and function data for ESM2-35M at https://huggingface.co/datasets/ktalreja/ProtoMechData/blob/main/functions_35M.tar.gz
(unzip first).