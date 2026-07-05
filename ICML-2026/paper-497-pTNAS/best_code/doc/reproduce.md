# pTNAS: Reproducing Key Tables

## Overview

pTNAS is a progressive NAS approach tailored for tabular data. It uses a filter-and-refine strategy:

1. **Coarse-Grained Filtering**: Evolutionary Algorithm + pTProxy (zero-cost proxy) to find top-K candidates
2. **Fine-Grained Refinement**: Successive Halving to select the best architecture
3. **Budget-Aware Coordinator**: allocates time between the two phases

Precomputed paper-side scripts and compact result files can be downloaded from [Google Drive](https://drive.google.com/file/d/1JpvzcwgPaw4TWVa1Ajmdo8SwFLPldFW-/view?usp=sharing) as `ptnas_run_outputs_20260517.tar.gz` and extracted under `run_outputs/`.

## Project Structure

```text
pTNAS/
├── src/
│   ├── common/              # Constants
│   ├── model/               # torch_frame encoder configs
│   ├── proxies/             # Zero-cost proxy evaluators
│   ├── search_algorithm/    # Evolutionary Algorithm
│   ├── search_space/        # MLP, ResDNN, BlockMixed
│   └── utils/               # TableData loading
├── utils/
│   └── query_api.py         # NAS-Bench-Tabular ground truth & proxy score query
├── scripts/
│   ├── nas_bench_exp.py     # NAS-Bench-Tabular Table 1 / anytime checks
│   ├── ptnas_full.py        # RelBench pTNAS implementation
│   ├── run_ptnas_full_time_budget.sh
│   ├── relbench/            # RelBench baseline runners
│   ├── nas_bench_tabular/   # NAS-Bench-Tabular baseline runners
│   └── new_space/           # ResDNN / BlockMixed training, scoring, analysis
├── run_outputs/
│   ├── README.md
│   ├── code/                # Paper-side aggregation and plotting scripts, from archive
│   └── data/                # Compact paper-side result files, from archive
├── datasets/
│   ├── nas_bench_tabular/   # Large NAS-Bench-Tabular benchmark data
│   └── fit-medium-table/    # RelBench medium-table data for rerunning raw methods
├── doc/
│   └── reproduce.md         # This reproduction guide
└── logs/                    # Temporary ad-hoc logs only, not paper result storage
```

## Code Entry Points

| Purpose | Entry point | Main inputs / outputs |
|---|---|---|
| Table 1 SRCC | `scripts/nas_bench_exp.py` | reads `datasets/nas_bench_tabular/` through `utils/query_api.py` |
| RelBench pTNAS | `scripts/ptnas_full.py`, `scripts/run_ptnas_full_time_budget.sh` | writes pTNAS CSV/logs under `run_outputs/data/relbench/ptnas/` |
| RelBench summaries | `run_outputs/code/compute_table2_summary.py`, `draw_figure4_performance.py`, `draw_figure5_efficiency.py` | reads `run_outputs/data/relbench/summary/` |
| NAS-Bench figures | `run_outputs/code/draw_figure*_*.py` | reads `datasets/nas_bench_tabular/` and `run_outputs/data/nas_bench_tabular/` |
| New-space correlation | `scripts/new_space/analysis/compute_resdnn_proxy_srcc.py`, `compute_blockmixed_proxy_srcc.py` | reads `datasets/nas_bench_tabular/{space_resdnn,space_blockmixed}/` |
| New-space search figure | `scripts/new_space/analysis/simulate_search_performance.py`, `plot_simulated_search_performance.py` | writes `run_outputs/data/new_space/` |

## Dataset

### NAS-Bench-Tabular (Table 1)

Pre-computed benchmark data (CSV files):

Download from: [Google Drive](https://drive.google.com/file/d/1Be1M0KD7z_YbyrhYSu9_lcfy5_NTueSd/view?usp=sharing)

After downloading, extract the archive and place `nas_bench_tabular/` under:

- `datasets/nas_bench_tabular/`

The data should include:

- `datasets/nas_bench_tabular/space_mlp/training/`
- `datasets/nas_bench_tabular/space_mlp/proxy_score/baseline/`
- `datasets/nas_bench_tabular/space_mlp/proxy_score/ptproxy/`

These files include:

- **Ground truth**: fully trained architectures on Frappe, UCI Diabetes, and Criteo
- **Proxy scores**: baseline zero-cost proxies and pTProxy scores

Then set the paths in `utils/query_api.py`:

- `_TRAIN_BASE`: path to `space_mlp/training/` (ground truth)
- `_PROXY_BASE`: path to `space_mlp/proxy_score/` (baseline + pTProxy scores)

In this repository, the default setup expects the benchmark data under `datasets/nas_bench_tabular/`.

### Paper-Side Result Files

The compact inputs used to reproduce the paper tables and figures are stored under:

- `run_outputs/data/`

These files are distributed as a separate script-result archive:

```text
https://drive.google.com/file/d/1JpvzcwgPaw4TWVa1Ajmdo8SwFLPldFW-/view?usp=sharing
```

Extract from the repository root:

```bash
tar -xzf run_outputs/ptnas_run_outputs_20260517.tar.gz -C run_outputs
```

Current local `run_outputs/data/` contains the final summarized RelBench tables, selected raw pTNAS / TabNAS / EA-NAS result files, the precomputed NAS-Bench-Tabular search traces used by Figures 3, 8, and 9, and compact result extracts for the new ResDNN / BlockMixed response experiments. Old exploratory logs are intentionally not kept there.

Current data tree:

```text
run_outputs/data/
├── relbench/
│   ├── summary/             # Table 11/12 CSVs and Figure 4 input
│   ├── ptnas/               # final pTNAS time-budget CSV/log
│   └── baselines/
│       └── nas/             # current TabNAS and EA-NAS CSVs
├── nas_bench_tabular/
│   ├── nas_ptnas/           # precomputed pTNAS anytime traces
│   ├── nas_re/              # EA anytime traces
│   ├── nas_tabnas/          # TabNAS anytime traces
│   ├── nas_rs/
│   ├── nas_rl/
│   ├── nas_bohb/
│   ├── nas_p1_examine/
│   └── ablation_study/      # Table 7-9 ablations and Figure 12/13 coordinator cache
└── new_space/
    ├── resnet_pool_proxy_srcc_summary.csv
    ├── blockmixed_selected_proxy_srcc_summary.csv
    └── simulated_search_performance_5seeds_detail.csv
```

If raw RelBench baseline logs are regenerated, the scripts write them under `run_outputs/data/relbench/baselines/{classical,deep_tabular,tabpfn,ltm}/`. Those rerun-only folders may be absent in a clean checkout until the corresponding baseline scripts are executed.

Use these paths as the first source when redrawing paper figures or recomputing paper summaries:

- `run_outputs/data/relbench/summary/summary_table15_all_results.csv`: Table 11 and Table 2 input
- `run_outputs/data/relbench/summary/summary_table16_timing.csv`: Table 12 and Figure 5 input
- `run_outputs/data/relbench/summary/summary_average_rankings_data.csv`: Figure 4 input
- `run_outputs/data/relbench/ptnas/`: final pTNAS time-budget run CSV/log
- `run_outputs/data/relbench/baselines/{tabnas_relbench_results.csv,renas_relbench_results.csv}`: TabNAS and EA-NAS raw result CSVs
- `run_outputs/data/relbench/baselines/{classical,deep_tabular,tabpfn,ltm}/`: raw logs/results produced only after rerunning those baselines
- `run_outputs/data/nas_bench_tabular/`: precomputed NAS-Bench traces and appendix result caches
- `datasets/nas_bench_tabular/{space_resdnn,space_blockmixed}/`: ResDNN / BlockMixed training results and proxy scores
- `run_outputs/data/new_space/`: ResDNN / BlockMixed SRCC summaries and simulated search trace

The larger benchmark data under `datasets/nas_bench_tabular/` are limited to benchmark ground-truth training results and proxy scores. They are required for Table 1, Figures 6/7, Figures 14-18, Tables 7-9, and for regenerating NAS-Bench-Tabular traces.

## Evaluation

First create and activate the conda environment:

```bash
conda create -n ptnas python=3.10 -y
conda activate ptnas
pip install -r requirements.txt
```

### Table 1: Correlation Benchmark (SRCC of zero-cost proxies)

```bash
cd pTNAS
export PYTHONPATH=src:.

# Read pre-computed scores and compute SRCC
python scripts/nas_bench_exp.py --exp srcc --dataset all

# Optional: compute pTProxy online for a subset of architectures
python scripts/nas_bench_exp.py --exp ptproxy_srcc --dataset frappe --max_archs 5000
```

The data are read from the local benchmark files in `datasets/nas_bench_tabular/`.

#### Figure 3: Progressive NAS Curves

Figure 3 is drawn from precomputed anytime search traces stored under:

- `run_outputs/data/nas_bench_tabular/nas_ptnas/`
- `run_outputs/data/nas_bench_tabular/nas_re/`
- `run_outputs/data/nas_bench_tabular/nas_tabnas/`

Use the following command to redraw the figure:

```bash
# Figure 3 plotting
# outputs: run_outputs/figure/anytime_{frappe,uci_diabetes,criteo}.pdf
python run_outputs/code/draw_figure3_anytime.py --dataset all
```

If needed, the baseline JSON files can be regenerated with:

```bash
# EA-NAS anytime traces
# outputs: run_outputs/data/nas_bench_tabular/nas_re/train_base_line_re_<dataset>_epoch_<N>.json
PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset frappe
PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset uci_diabetes
PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset criteo

# TabNAS anytime traces
# outputs: run_outputs/data/nas_bench_tabular/nas_tabnas/tabNAS_benchmark_<dataset>_epoch_<N>.json
PYTHONPATH=src python scripts/nas_bench_tabular/tab_nas.py --dataset frappe
PYTHONPATH=src python scripts/nas_bench_tabular/tab_nas.py --dataset uci_diabetes
PYTHONPATH=src python scripts/nas_bench_tabular/tab_nas.py --dataset criteo

# pTNAS anytime evaluation
# output: printed anytime results in the terminal
# note: the committed run_outputs/data/nas_bench_tabular/nas_ptnas/*.json files are precomputed
# anytime traces used directly by draw_figure3_anytime.py; the current
# nas_bench_exp.py --exp anytime command does not automatically rewrite them.
python scripts/nas_bench_exp.py --exp anytime --dataset frappe
python scripts/nas_bench_exp.py --exp anytime --dataset uci_diabetes
python scripts/nas_bench_exp.py --exp anytime --dataset criteo
```

### Effectiveness on RelBench

RelBench data are expected under:

- `datasets/fit-medium-table/`

The final paper-side RelBench inputs are already summarized in:

- `run_outputs/data/relbench/summary/summary_table15_all_results.csv`
- `run_outputs/data/relbench/summary/summary_table16_timing.csv`
- `run_outputs/data/relbench/summary/summary_average_rankings_data.csv`

The current final pTNAS raw run is kept in:

- `run_outputs/data/relbench/ptnas/ptnas_full_timebudget_10s_20260512_212458.csv`
- `run_outputs/data/relbench/ptnas/run_ptnas_full_timebudget_10s_20260512_212458.log`

If needed, rerun the following methods to regenerate raw per-dataset outputs. Generated logs and CSVs are written under `run_outputs/data/relbench/`.

```bash
# pTNAS
# current Table 11 / Table 12 pTNAS run
# outputs: run_outputs/data/relbench/ptnas/ptnas_full_timebudget_<time>s_<timestamp>.csv,
#          run_outputs/data/relbench/ptnas/run_ptnas_full_timebudget_<time>s_<timestamp>.log
bash scripts/run_ptnas_full_time_budget.sh

# Logistic Regression / Random Forest
# rerun outputs: run_outputs/data/relbench/baselines/classical/logs/
bash scripts/relbench/sklearn_baseline.sh

# LightGBM / CatBoost
# rerun outputs: run_outputs/data/relbench/baselines/classical/logs/
bash scripts/relbench/ml_baseline.sh

# DNN (MLP) / DeepFM (DFM) / FT-Transformer (FTTrans) / ARM-Net (ARMNet)
# rerun outputs: run_outputs/data/relbench/baselines/deep_tabular/dnn_results.csv
#                run_outputs/data/relbench/baselines/deep_tabular/dnn_<model>_<dataset>_<timestamp>.log
bash scripts/relbench/dnn_baseline.sh

# TabPFN
# rerun outputs: run_outputs/data/relbench/baselines/tabpfn/logs/
bash scripts/relbench/tabpfn_baseline.sh

# TabNAS
# current CSV: run_outputs/data/relbench/baselines/tabnas_relbench_results.csv
# rerun logs:  run_outputs/data/relbench/baselines/tabnas_logs/
bash scripts/relbench/nas/run_tabnas_relbench.sh

# EA-NAS
# current CSV: run_outputs/data/relbench/baselines/renas_relbench_results.csv
# rerun logs:  run_outputs/data/relbench/baselines/renas_logs/
bash scripts/relbench/nas/run_renas_relbench.sh

# TP-BERTa / Nomic / BGE
# rerun outputs: run_outputs/data/relbench/baselines/ltm/tpberta_table/
#                run_outputs/data/relbench/baselines/ltm/results/
bash scripts/relbench/LTM/scripts/save_medium_embed_csv.sh
bash scripts/relbench/LTM/scripts/train_ltm.sh

# TabICL
# script/notebook: scripts/relbench/test_tabicl.ipynb
# outputs: notebook output / manual logs
```

After collecting or updating raw outputs, consolidate them into the local summary files used by the paper:

- `run_outputs/data/relbench/summary/summary_table15_all_results.csv`: per-dataset predictive results (Table 11 style)
- `run_outputs/data/relbench/summary/summary_table16_timing.csv`: per-dataset timing results (Table 12 / Figure 5 style)

Then run the paper-side analysis:

```bash
# Table 11
# input: run_outputs/data/relbench/summary/summary_table15_all_results.csv

# Figure 4
# input: run_outputs/data/relbench/summary/summary_average_rankings_data.csv
# figure outputs: run_outputs/figure/performance_scatter_combined.pdf
python run_outputs/code/draw_figure4_performance.py

# Table 12 / Figure 5
# input: run_outputs/data/relbench/summary/summary_table16_timing.csv
# figure outputs: run_outputs/figure/runtime_task*_avg.pdf
python run_outputs/code/draw_figure5_efficiency.py

# Table 2
# input: run_outputs/data/relbench/summary/summary_table15_all_results.csv
# output: printed Table 2 summary only
python run_outputs/code/compute_table2_summary.py
```

### Appendix

#### Figure 6/7: Search Space Characterization

Figures 6 and 7 are drawn directly from the NAS-Bench-Tabular ground-truth CSV files under:

- `datasets/nas_bench_tabular/space_mlp/training/`

```bash
# Figure 6: ECDF of train/validation AUC
# Figure 7: parameter size vs. validation AUC
# outputs: run_outputs/figure/ecdf_<dataset>.pdf,
#          run_outputs/figure/param_vs_auc_<dataset>.pdf
python run_outputs/code/draw_figure6_7_space.py --dataset all
```

#### Figure 8: Search Strategy Benchmark

Figure 8 compares RS, RL, EA, and BOHB using precomputed NAS-Bench-Tabular search traces under:

- `run_outputs/data/nas_bench_tabular/nas_rs/`
- `run_outputs/data/nas_bench_tabular/nas_rl/`
- `run_outputs/data/nas_bench_tabular/nas_re/`
- `run_outputs/data/nas_bench_tabular/nas_bohb/`

```bash
# Figure 8: benchmarking search strategies
# outputs: run_outputs/figure/benchmark_<dataset>.pdf
python run_outputs/code/draw_figure8_search_strategies.py --dataset all
```

If needed, regenerate the search traces with:

```bash
# RS traces
# outputs: run_outputs/data/nas_bench_tabular/nas_rs/train_base_line_rs_<dataset>_epoch_<N>.json
PYTHONPATH=src python scripts/nas_bench_tabular/rs_nas.py --dataset frappe
PYTHONPATH=src python scripts/nas_bench_tabular/rs_nas.py --dataset uci_diabetes
PYTHONPATH=src python scripts/nas_bench_tabular/rs_nas.py --dataset criteo

# RL traces
# outputs: run_outputs/data/nas_bench_tabular/nas_rl/train_base_line_rl_<dataset>_epoch_<N>.json
PYTHONPATH=src python scripts/nas_bench_tabular/rl_nas.py --dataset frappe
PYTHONPATH=src python scripts/nas_bench_tabular/rl_nas.py --dataset uci_diabetes
PYTHONPATH=src python scripts/nas_bench_tabular/rl_nas.py --dataset criteo

# EA traces
# outputs: run_outputs/data/nas_bench_tabular/nas_re/train_base_line_re_<dataset>_epoch_<N>.json
PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset frappe
PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset uci_diabetes
PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset criteo

# BOHB traces
# outputs: run_outputs/data/nas_bench_tabular/nas_bohb/train_base_line_bohb_<dataset>_epoch_<N>.json
# note: this script requires hpbandster and ConfigSpace.
PYTHONPATH=src python scripts/nas_bench_tabular/bohb_nas.py --dataset frappe
PYTHONPATH=src python scripts/nas_bench_tabular/bohb_nas.py --dataset uci_diabetes
PYTHONPATH=src python scripts/nas_bench_tabular/bohb_nas.py --dataset criteo
```

#### Figure 9 / Table 6 / Figure 10: New Search Space

The compact ResDNN and BlockMixed architecture lists, training results, and proxy scores are stored under `datasets/nas_bench_tabular/{space_resdnn,space_blockmixed}/`. Derived SRCC summaries and simulated search traces are stored under `run_outputs/data/new_space/`.

Figure 9 is a static BlockMixed search-space schematic rather than a result computed by a Python script. Table 6 is computed from the SRCC summary scripts below. Figure 10 is redrawn from the simulated search-performance detail CSV.

- `datasets/nas_bench_tabular/space_resdnn/architecture/random_sampled_arch_resdnn_{classification,regression}.txt`: sampled ResDNN architecture lists
- `datasets/nas_bench_tabular/space_blockmixed/architecture/blockmixed.txt`: sampled BlockMixed architecture list
- `datasets/nas_bench_tabular/space_blockmixed/architecture/random_sampled_arch_blockmixed_metadata.json`: dataset-specific BlockMixed group selections
- `datasets/nas_bench_tabular/space_resdnn/training/resnet_pool_results.csv`: fully trained ResDNN architecture results
- `datasets/nas_bench_tabular/space_blockmixed/training/block_mixed_diverse_results.csv`: fully trained BlockMixed architecture results
- `space_{resdnn,blockmixed}/proxy_score/ptproxy/`: finalized pTProxy `v1` score CSVs
- `space_{resdnn,blockmixed}/proxy_score/baseline/`: merged baseline proxy score CSVs, one `score_<dataset>.csv` per dataset; the `proxy` column identifies Fisher, SNIP, NASWOT, and the other baseline methods
- `run_outputs/data/new_space/*_proxy_srcc_summary.csv`: computed proxy-comparison SRCC summaries
- `run_outputs/data/new_space/simulated_search_performance_5seeds_detail.csv`: simulated search-performance trace

1. Correlation

Train the architecture pools, score each proxy, merge proxy scores into one CSV per dataset, then compute SRCC summaries.

```bash
# Ground-truth trained architecture results
# outputs:
#   datasets/nas_bench_tabular/space_resdnn/training/resnet_pool_results.csv
#   datasets/nas_bench_tabular/space_blockmixed/training/block_mixed_diverse_results.csv
bash scripts/new_space/training/run_train_resdnn.sh
bash scripts/new_space/training/run_train_blockmixed.sh

# Proxy scores
# outputs:
#   datasets/nas_bench_tabular/space_resdnn/proxy_score/{ptproxy,baseline}/
#   datasets/nas_bench_tabular/space_blockmixed/proxy_score/{ptproxy,baseline}/
bash scripts/new_space/proxy_scores/run_score_resdnn_proxies.sh
bash scripts/new_space/proxy_scores/run_score_resdnn_classic.sh
bash scripts/new_space/proxy_scores/run_score_blockmixed_ptproxy.sh
bash scripts/new_space/proxy_scores/run_score_blockmixed_classic.sh
```

The scoring scripts save baseline proxy scores as `datasets/nas_bench_tabular/<space>/proxy_score/baseline/score_<dataset>.csv`.

```bash
# Proxy-comparison SRCC for ResDNN and BlockMixed
# outputs:
#   run_outputs/data/new_space/resnet_pool_proxy_srcc_summary.csv
#   run_outputs/data/new_space/blockmixed_selected_proxy_srcc_summary.csv
python scripts/new_space/analysis/compute_resdnn_proxy_srcc.py
python scripts/new_space/analysis/compute_blockmixed_proxy_srcc.py
```

2. Search Performance

The search-performance figure is generated with an offline simulator. It does not rerun GPU training. For each seed, the simulator samples nested architecture prefixes for `M=5,10,20,40,...`, ranks candidates by saved pTProxy scores, keeps `topK=ceil(M/30)`, simulates SH selection with saved validation metrics, and reports the selected architecture's saved test metric. The figure plots the five-seed mean best-so-far performance.

```bash
# pTNAS search performance over five random seeds.
# outputs:
#   run_outputs/data/new_space/simulated_search_performance_5seeds_detail.csv
python scripts/new_space/analysis/simulate_search_performance.py \
  --seeds 42,43,44,45,46 \
  --output_csv run_outputs/data/new_space/simulated_search_performance_5seeds_detail.csv

# Draw the four search-performance panels and standalone legend.
# The plotting script computes the five-seed mean from the detail CSV.
# outputs:
#   run_outputs/data/new_space/search_performance_figs/*.pdf
python scripts/new_space/analysis/plot_simulated_search_performance.py
```

#### Figure 11: pTProxy Score vs. AUC

Figure 11 is drawn from precomputed coarse-filtering traces under:

- `run_outputs/data/nas_bench_tabular/nas_p1_examine/`

```bash
# Figure 11: pTProxy score and searched AUC vs. explored architectures
# outputs: run_outputs/figure/p1_score_auc_<dataset>.pdf
python run_outputs/code/draw_figure11_p1_examine.py --dataset all
```

To regenerate the traces:

```bash
# Phase-1 examine traces
# outputs: run_outputs/data/nas_bench_tabular/nas_p1_examine/re_<dataset>_-1_12_auc.json
#          run_outputs/data/nas_bench_tabular/nas_p1_examine/re_<dataset>_-1_12_score.json
# note: this step queries ground-truth AUC from NAS-Bench-Tabular and computes pTProxy scores online.
PYTHONPATH=src python scripts/nas_bench_tabular/p1_examine.py --dataset all
```

#### Figure 12/13: Coordinator Analysis

Figure 12 draws the K/U heatmaps. Figure 13 draws the M/K trade-off curves from the available coordinator JSON files under:

- `run_outputs/data/nas_bench_tabular/ablation_study/coordinator/`

```bash
# Figure 12: K/U heatmaps
# Figure 13: M/K trade-off curves
# outputs: run_outputs/figure/micro_ku_*.pdf,
#          run_outputs/figure/trade_off_nk_<dataset>.pdf
python run_outputs/code/draw_figure12_13_coordinator.py
```

The current local coordinator JSON files regenerate the Frappe and Criteo Figure 13 panels. The Diabetes panel included in the paper figure directory is a precomputed panel.

#### Figure 14-18: Proxy Correlation Scatter Plots

Figures 14-18 are drawn from the NAS-Bench-Tabular ground-truth results, baseline proxy scores, and pTProxy scores:

- `datasets/nas_bench_tabular/space_mlp/training/`
- `datasets/nas_bench_tabular/space_mlp/proxy_score/baseline/`
- `datasets/nas_bench_tabular/space_mlp/proxy_score/ptproxy/`

The plotting script samples 4000 common architectures per dataset and proxy, then plots proxy score against validation AUC.

```bash
# Figure 14-18: proxy score vs. validation AUC scatter plots
# outputs: run_outputs/figure/proxy_scatter_*.pdf
python run_outputs/code/draw_figure14_18_proxy_scatter.py
```

#### Table 7-9: pTProxy Sensitivity Ablations

Tables 7-9 are computed from the local NAS-Bench-Tabular sensitivity files:

- `run_outputs/data/nas_bench_tabular/ablation_study/`
- `datasets/nas_bench_tabular/space_mlp/training/`

The script computes SRCC with `scipy.stats.spearmanr` against the ground-truth validation AUC.

```bash
# Table 7: parameter positivity
# Table 8: initialization method
# Table 9: batch size
# outputs: printed table values only
python run_outputs/code/compute_table7_8_9.py
```

#### Table 10: Recalibration Weight Ablation

Table 10 compares three pTProxy recalibration variants: width only, depth only, and the default width-depth recalibration.

Unlike Tables 7-9, this script computes pTProxy scores online, so it is slower. Use `--max_archs` for a faster partial check.

```bash
# Table 10: recalibration weights
# outputs: printed SRCC values only
PYTHONPATH=src:. python run_outputs/code/compute_table10_recalibration.py

# Optional faster check
PYTHONPATH=src:. python run_outputs/code/compute_table10_recalibration.py --max_archs 5000
```
