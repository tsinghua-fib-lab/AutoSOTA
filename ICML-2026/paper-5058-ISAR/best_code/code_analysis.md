# SOTA Preparation Repair Analysis — Paper 5058

## Preparation Failure

The SOTA preparation failed for two reasons in the reusable `autosota_repro_paper_5058` container:
1. **git not installed**: The `autosota/paper-5058:env` base image lacked git.
2. **dpkg lock contention**: The first docker exec attempt hit a concurrent apt-get process.

When a new container `autosota_sota_paper_5058` was started from `autosota/paper-5058:reproduced`, it also lacked git, and dpkg needed `--configure -a`.

### Repair Steps
1. Fixed dpkg state: `dpkg --configure -a`
2. Installed git: `apt-get install -y git`
3. Created `/tools/record_score.sh` from host script
4. Initialized git repo at `/repo`, created baseline commit and `_baseline` tag
5. Verified evaluation reproduces baseline: ratio 1.6771 (1127/672)

## Corrected Evaluation Command

```bash
cd /repo && python3 evaluate.py
```

This runs inside `autosota_sota_paper_5058` container and:
1. Runs ISAR binary (`/repo/build/ISAR`) to compute MWU lower bound
2. Runs SCC evolutionary binary (`scc_evolutionary_int`) with multiple seeds for upper bound
3. Computes ISAR ratio = SCC_disagreements / MWU_lower_bound
4. Outputs JSON with all metrics to stdout

## Baseline Verification
- MWU lower bound (eps=0.05): **672**
- Best SCC upper bound (disagreements): **1127** (4 seeds, 30s each)
- ISAR ratio: **1.6771** — matches reproduction manifest exactly
- Paper reported ratio: 1.708

## Reusable Resources

### Datasets
- `/datasets/bitcoinotc_dedup.cc` — Deduplicated bitcoinOTC dataset (21492 edges, 5881 nodes)
- `/datasets/bitcoinotc_from_cc.graph` — Corrected .graph file matching .cc edge signs
- `/datasets/bitcoinotc.graph` — Original .graph (358 sign mismatches — do not use)
- `/datasets/bitcoinotc.cc` — Original .cc file (do not use for evaluation)

### Binaries
- `/repo/build/ISAR` — ISAR binary (MWU lower bound computation)
- `/autosota_cache/ScalableCorrelationClustering/build/scc_evolutionary_int` — SCC evolutionary solver

### Cache
- `/autosota_cache/` — Build artifacts, SCC source, pre-downloaded data

## Safe Optimization Targets

1. **SCC solver parameters**: seed count, time_limit — safe, no metric change
2. **Post-processing**: local search refinement of SCC clustering — safe, no metric change
3. **Initialization**: spectral/MWU-guided seeds for SCC — safe, no metric change
4. **MWU parameters**: epsilon, rho — valid certificate changes, compute ratio with matching MWU
5. **SCC preconfiguration**: different configs (strong, ssocial) — safe, no metric change
6. **Graph preprocessing**: edge pruning, positive subgraph contraction — must run MWU+SCC on same graph

## Red Lines
- Do NOT change the .cc disagreement counting logic
- Do NOT change the MWU lower bound certificate verification
- Do NOT change the ratio formula
- All improvements must use the same ratio definition as baseline

## Container Access
- Container: `autosota_sota_paper_5058`
- Repo: `/repo`
- Artifacts: `/autosota_artifacts/paper-5058/sota/`
- Tools: `/tools/record_score.sh`
