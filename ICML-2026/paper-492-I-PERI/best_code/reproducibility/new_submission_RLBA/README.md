# RBAL experiments

This folder contains the evaluation script used to benchmark the temporal difference-graph algorithms and the baselines on synthetic time-series data generated.

The main entry point is:

```bash
experiments.py
```

The script generates normal and anomalous regimes, runs the selected algorithms, scores their predictions against the known changed mechanisms, and writes summary tables with the mean, standard deviation, and variance of the F1 score.

---

## Expected project layout

Current development layout:

```text
RBAL/
├── generator.py
│── experiments.py
│── baseline/
│    ├── MBGH.py
│    ├── microcause.py
│    ├── estimation.py
│    ├── rcd.py
│    └── utils_rcd.py
```

---

## What `experiments.py` does

For each selected setting, number of nodes, sample size, and repetition, the script:

1. Generates one synthetic run using `generate_one_run` from `generator.py`.
2. Extracts the ground-truth changed edges and shifted nodes.
3. Runs the selected algorithms.
4. Scores graph algorithms with `evaluate_all_ts`.
5. Scores node-ranking algorithms such as MicroCause and RCD against the true shifted nodes.

The graph algorithms are evaluated using edge/node metrics defined in `tsalgos/metrics_ts.py`.

---

## Available algorithms

The active algorithms are:

```text
tsldiffpc      Temporal linear difference PC
tsldiffpc_pc   tsLDiffPC with additional tPC orientation
tsdci          Temporal DCI
tsdci_pc       tsDCI with additional tPC orientation
tsMBGH         MBGH baseline
microcause     MicroCause root-cause baseline
rcd            RCD root-cause baseline
```

`tsiSCAN` is intentionally disabled for now.

---

## Settings

The default benchmark settings are:

```text
setting1_lag2
setting2_lag1
setting3_contemporaneous_with_self_lag
setting4_iid
```

Default user lags are:

```python
setting1_lag2: [2]
setting2_lag1: [1, 2]
setting3_contemporaneous_with_self_lag: [0, 1, 2]
setting4_iid: [0, 1, 2]
```

---

## Results

To reproduce the results of the paper run :

```bash
python RBAL/experiments.py \
  --settings setting2_lag1 \
  --p-list 3 \
  --n-list 1000 \
  --n-reps 10 \
  --user-lags 1 \
  --change-model single_edge all_parents all_parents_min2 \
  --algos tsldiffpc tsldiffpc_pc tsdci tsdci_pc tsMBGH microcause rcd
```
