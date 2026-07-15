# ProbeSwitch: Transfer + Overhead Summary

This file contains two compact tables.

## A) Threshold transfer (zero tuning)

Transfer rule shown below: `bbob_B500` (transfer (COCO-learned, B=500D), t=0.12).

Safe rule: `fixed0p22` (safe transfer (fixed t=0.22), t=0.22). Status `boundary` marks regret worse than always-CMA.

| target | transfer status | safe status | always-CMA regret | transfer t | transfer acc | transfer regret | safe t | safe acc | safe regret | tuned t | tuned acc | tuned regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO D40 B200 | ok | ok | 0.096 | 0.12 | 0.742 | 0.035 | 0.22 | 0.729 | 0.044 | 0.18 | 0.731 | 0.037 |
| COCO D40 B500 | ok | ok | 0.096 | 0.12 | 0.755 | 0.024 | 0.22 | 0.745 | 0.040 | 0.12 | 0.755 | 0.024 |
| LogReg (synth) | ok | ok | 0.087 | 0.12 | 0.773 | 0.011 | 0.22 | 0.727 | 0.019 | 0.01 | 0.773 | 0.011 |
| LogReg (BC) | ok | ok | 0.149 | 0.12 | 0.680 | 0.062 | 0.22 | 0.560 | 0.115 | 0.01 | 0.720 | 0.059 |
| LogReg (digits0) | ok | ok | 0.126 | 0.12 | 0.640 | 0.045 | 0.22 | 0.587 | 0.054 | 0.00 | 0.680 | 0.036 |
| MLP (digits0, HT) | ok | ok | 0.177 | 0.12 | 0.636 | 0.073 | 0.22 | 0.500 | 0.124 | 0.01 | 0.636 | 0.073 |
| LQR (HT) | ok | boundary | 0.054 | 0.12 | 0.480 | 0.052 | 0.22 | 0.360 | 0.061 | 0.09 | 0.560 | 0.036 |
| HPO (digits0) | boundary | boundary | 0.041 | 0.12 | 0.560 | 0.052 | 0.22 | 0.660 | 0.043 | 0.22 | 0.660 | 0.043 |
| RL (CartPole) | boundary | boundary | 0.153 | 0.12 | 0.400 | 0.343 | 0.22 | 0.520 | 0.271 | 0.28 | 0.640 | 0.166 |

## B) VOI / overhead-vs-gain curve (logreg sweep)

Lower is better. `bs=8` is stochastic/high-misranking; `bs=256` is deterministic.

| batch | B/d | median(CMA) | median(Switch) | median(Warmstart) | p(CMA vs Switch) | p(CMA vs Warmstart) | p(Switch vs Warmstart) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 20 | 5.507 | 4.691 | 4.674 | 0.0066 | 0.0026 | 1.0000 |
| 8 | 40 | 5.068 | 4.114 | 4.069 | 5.6e-06 | 1.2e-06 | 1.0000 |
| 8 | 80 | 4.925 | 3.800 | 3.800 | 2.1e-07 | 2.1e-07 | - |
| 256 | 20 | 1.051 | 1.089 | 1.056 | 4.7e-10 | 0.0039 | 9.3e-10 |
| 256 | 40 | 0.549 | 0.562 | 0.549 | 1.1e-13 | 0.0156 | 2.3e-13 |
| 256 | 80 | 0.371 | 0.372 | 0.371 | 3.6e-12 | 0.0020 | 5.8e-11 |
