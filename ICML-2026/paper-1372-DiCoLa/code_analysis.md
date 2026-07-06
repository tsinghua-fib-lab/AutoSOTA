# SOTA Preparation Repair — Paper 1372: DiCoLA

## Original Failure

The normal SOTA preparation path failed because:

1. **git not installed**: The PyTorch 2.1.0 runtime Docker image does not include `git`.
2. **Network proxy blocked apt**: The container was configured with HTTP_PROXY pointing to `172.17.0.1:17890`, but `apt-get` does not use environment proxy variables. The proxy also appeared to reject connections (`Connection failed`).
3. **conda proxy failure**: Similarly, `conda install` failed because the proxy configuration was broken.

## Repair Applied

1. **Unset proxy for package management**: Running `unset` on all proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`, `http_proxy`, `https_proxy`, `ALL_PROXY`, `all_proxy`) allowed direct package downloads through the Tsinghua mirror and Ubuntu archive.
2. **Install git**: `apt-get install -y git` succeeded without proxy.
3. **Create /tools/record_score.sh**: Copied from host `/home/dataset-assist-0/chenzhibin/autosota-v2.5-1/auto_sota/agents/sota/scripts/record_score.sh` into container.
4. **Git repo setup**: Initialized git in `/repo`, created baseline commit, tagged `_baseline`.

## Corrected In-Container Evaluation Command

```bash
cd /repo && python3 experiment_er.py
```

No proxy environment variables should be set. The evaluation runs 50 trials of ER(50,3) graph generation with 5 latent variables, 2000 samples, and alpha=0.01, comparing FCI and DiCoLA+FCI.

## Baseline Verification

The baseline was verified against the reproduction manifest:
- CI_Tests_DiCoLA: 6302.9 ✓
- F1_DiCoLA: 0.71 ✓
- Precision_DiCoLA: 0.93 ✓
- Recall_DiCoLA: 0.58 ✓
- Time_DiCoLA: 0.90 ✓

## Reusable /paper_data Resources

The `/paper_data` mount contains:
- `DiCoLA-ICML2026/`: Source code (duplicate of /repo)
- `causal-learn-pkg/`: Full causal-learn package (vendored version used by code)
- `causality-lab/`: Related causality tools
- `rcd/`: RCD (Rank-based Causal Discovery) package
- Various `.rda` and `.csv` data files (not used by the ER experiment)

The ER experiment generates synthetic data and does not use any pre-downloaded datasets.

## Safe Optimization Targets

The optimization targets were the algorithmic parameters in four files:
1. `DiCoLa/CI_test.py` — CI test cache size, numerical stability
2. `DiCoLa/Recursive_PAG.py` — min_leaf_size, max_recursion_depth, decomposition scoring
3. `compare_algs/causallearn_package/utils/FAS.py` — FCI adjacency search (depth loop, conditioning sets)
4. `experiment_er.py` — parameter passthrough

All changes are to algorithm internals (cache sizes, parameter defaults, search heuristics) and do NOT modify the evaluation protocol, metrics, test data, or benchmark outputs.
