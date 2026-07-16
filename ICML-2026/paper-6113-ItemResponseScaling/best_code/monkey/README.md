# Test-Time Scaling Experiments

This directory contains the complete pipeline for test-time scaling experiments using Item Response Theory.

## Directory Structure Overview

The paper uses **TWO** test-time datasets (Section 4.2: $P^{\text{testtime}}_1$ and $P^{\text{testtime}}_2$):

```
monkey/
├── monkey_query/          → Query LLMs for Dataset 1 (ResMAT1)
├── monkey_gather_data/    → Process queries into ResMAT1
│
├── monkey_query_2/        → Query LLMs for Dataset 2 (ResMAT2)
├── monkey_gather_data_2/  → Process queries into ResMAT2
│
└── monkey_analysis/       → Analyze BOTH datasets, generate paper figures
```

**Important**: The `_2` suffix does NOT mean "version 2" - it means "Dataset 2" as described in the paper!

---

## Dataset Specifications

### ResMAT1 ($P^{\text{testtime}}_1$)

| Property | Value |
|----------|-------|
| **Models** | 8 LLMs |
| **Questions** | 552 from 6 benchmarks |
| **Samples/question** | 10,000 |
| **Prompting** | Few-shot, temp 1.0 |
| **Benchmarks** | commonsense, legalbench, med_qa, bbq, lsat_qa, legal_support |
| **Purpose** | HELM validation, z-score correlation |
| **HF Dataset** | `stair-lab/irsl_testtime_resmat1` |
| **Directories** | [`monkey_query/`](monkey_query/) + [`monkey_gather_data/`](monkey_gather_data/) |

### ResMAT2 ($P^{\text{testtime}}_2$)

| Property | Value |
|----------|-------|
| **Models** | 12 latest LLMs |
| **Questions** | 120 from 4 benchmarks |
| **Samples/question** | ~2,500-2,560 |
| **Prompting** | Zero-shot CoT, temp 0.6 |
| **Benchmarks** | AIME 2024, AIME 2025, MMLU Pro, Global MMLU Lite |
| **Purpose** | Recent models, main figures |
| **HF Dataset** | `stair-lab/irsl_testtime_resmat2` |
| **Directories** | [`monkey_query_2/`](monkey_query_2/) + [`monkey_gather_data_2/`](monkey_gather_data_2/) |

---

## Pipeline Flow

### For ResMAT1
```
1. monkey_query/query.py
   ↓ (generates 10k samples per question)
2. Upload to stair-lab/monkey_queries
   ↓
3. monkey_gather_data/testtime_gather_3d_resmat.py
   ↓ (downloads & processes)
4. Upload to stair-lab/irsl_testtime_resmat1
   ↓
5. monkey_analysis/testtime_calibrate.py
   ↓ (calibrates item difficulties)
6. monkey_analysis/testtime_cat.py
   ↓ (runs CAT)
7. Paper figures
```

### For ResMAT2
```
1. monkey_query_2/query.py
   ↓ (generates 2.5k samples per question)
2. Upload to stair-lab/denoise_eval_query
   ↓
3. monkey_gather_data_2/gather.py
   ↓ (downloads & processes with quality control)
4. Upload to stair-lab/irsl_testtime_resmat2
   ↓
5. monkey_analysis/testtime_calibrate.py
   ↓ (calibrates item difficulties)
6. monkey_analysis/testtime_cat.py
   ↓ (runs CAT)
7. Paper figures
```

---

## Quick Start

### Reproduce ResMAT1 Results
```bash
cd monkey/monkey_analysis

# The script automatically downloads stair-lab/irsl_testtime_resmat1
python testtime_calibrate.py  # Calibrate item difficulties
python testtime_cat.py         # Run CAT and generate figures
python testtime_aggregate_results.py  # Aggregate results
```

### Reproduce ResMAT2 Results
```bash
cd monkey/monkey_analysis

# Same scripts, they detect which dataset to use via FILE_NAME variable
# Edit scripts to set FILE_NAME = "irsl_testtime_resmat2"
python testtime_calibrate.py
python testtime_cat.py
python testtime_aggregate_results.py
```

### Generate New Queries (Advanced)

**For ResMAT1**:
```bash
cd monkey/monkey_query
python monkey_query.py --model_nickname <model> --dataset <benchmark> --num_samples 10000
```

**For ResMAT2**:
```bash
cd monkey/monkey_query_2
python query.py --model_nickname <model> --dataset <benchmark> --num_samples 2560

# Or use batch scripts
bash query_llama.sh
bash query_small.sh
```

---

## Analysis Scripts

All analysis scripts are in [`monkey_analysis/`](monkey_analysis/):

| Script | Purpose | Datasets |
|--------|---------|----------|
| `testtime_calibrate.py` | Calibrate item difficulties (z-scores) | Both |
| `testtime_cat.py` | Run CAT, generate convergence plots | Both |
| `testtime_aggregate_results.py` | Aggregate results, generate figures 5,8,9,10 | Both |
| `monkey_analysis_denoise_data.py` | Denoising analysis (Figure 6) | Special 3D data |

---

## Paper Figures

| Figure | Dataset Used | Script |
|--------|-------------|--------|
| Figure 5, 8 | Both ResMAT1 & ResMAT2 | `testtime_aggregate_results.py` |
| Figure 6 | Special 3D data | `monkey_analysis_denoise_data.py` |
| Figure 9, 10 | Both ResMAT1 & ResMAT2 | `testtime_aggregate_results.py` |
| Appendix z-correlation | ResMAT1 only | `testtime_cat.py` |

---

## Data Quality

### ResMAT1
- All 8 models included
- Complete 10,000 samples per question
- No exclusions

### ResMAT2
- 12 models excluded (incomplete data)
- 19 MMLU Pro questions excluded (cross-model incompleteness)
- See [`monkey_gather_data_2/README.md`](monkey_gather_data_2/README.md) for full exclusion lists

---

## Further Documentation

- ResMAT1 query details: [`monkey_query/README.md`](monkey_query/README.md)
- ResMAT1 gathering details: [`monkey_gather_data/README.md`](monkey_gather_data/README.md)
- ResMAT2 query details: [`monkey_query_2/README.md`](monkey_query_2/README.md)
- ResMAT2 gathering details: [`monkey_gather_data_2/README.md`](monkey_gather_data_2/README.md)
- Investigation report: [`../TESTTIME_DATA_INVESTIGATION.md`](../TESTTIME_DATA_INVESTIGATION.md)
- Changes summary: [`../CHANGES_SUMMARY.md`](../CHANGES_SUMMARY.md)
