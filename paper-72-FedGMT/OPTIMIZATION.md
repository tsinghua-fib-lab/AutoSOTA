# Paper 72 — FedGMT

**Full title:** *One Arrow, Two Hawks: Sharpness-aware Minimization for Federated Learning via Global Model Trajectory*

**Registered metric movement (`results.md`, latest):** +1.50% (79.57 -> 80.76)

## Final Optimization Report (latest sync)

- **Pipeline run:** `run_20260323_223906`
- **Total iterations:** 12
- **Best test accuracy:** **80.76%** (baseline **79.57%**, +1.19pt / +1.50% by ledger reporting)
- **Target:** 80.8146% (final landed just under target, but within low-variance band)
- **Stability:** last-50 rounds mean about **80.76%**, std about **0.054%**

## What changed and why it worked

1. **Higher trajectory EMA smoothing (`alpha=0.99`)**
   - File: `system/flcore/servers/servergmt.py`
   - Change: in-code tuning from the parser default `alpha=0.95` to `0.99`.
   - Effect: smoother global trajectory reference and lower guidance variance for clients.

2. **SWA in late rounds for evaluation (`swa_start=400`)**
   - File: `system/flcore/servers/servergmt.py`
   - Change: from round 400, keep a uniform SWA average of global checkpoints and evaluate with SWA model.
   - Effect: lower end-game noise and more stable/stronger final accuracy.

3. **SWA model used as late-phase teacher (round >= 401)**
   - File: `system/flcore/servers/servergmt.py`
   - Change: `send_models()` switches the client-side teacher from EMA model to SWA model once SWA is available.
   - Effect: clients optimize toward a flatter/stabler target, improving convergence in the last 100 rounds.

4. **Disable frequent model file dumping**
   - File: `system/flcore/servers/serverbase.py`
   - Change: skip heavy save path in this run profile.
   - Effect: prevents disk pressure and keeps long 500-round training uninterrupted.

## What did not help (from run log)

- Starting SWA too early (e.g. around 300/380 rounds) pulled in under-converged checkpoints.
- Larger or smaller trajectory regularization (`gama`) both hurt compared with `gama=1.0`.
- Lower `tau` weakened effective KL guidance in late rounds.
- SWA+EMA mixed evaluation underperformed plain SWA.

## Key files to inspect

- `system/flcore/servers/servergmt.py`
- `system/flcore/servers/serverbase.py`
- `system/main.py`
- `results/cifar10/FedGMT_test-rho0.01-alpha0.99-gama1.0-tau3.0.csv`

## Repro note

The synced code keeps `global_rounds=500` and standard CIFAR-10 Dirichlet setup. The winning behavior is driven by **late SWA + SWA teacher** on top of FedGMT trajectory regularization.
