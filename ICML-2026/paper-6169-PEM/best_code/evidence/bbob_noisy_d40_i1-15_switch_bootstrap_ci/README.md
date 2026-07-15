# Bootstrap CI: ProbeSwitch vs CMA-ES (COCO bbob-noisy)

Purpose: complement sign-test/Wilcoxon with an effect-size confidence interval on paired differences.

Metric: COCO noise-free final best (delta to fopt), aggregated over (budget,function,dimension,instance) pairs.

Effect definition: Δ = log10(best_f_A) - log10(best_f_B), so **negative** Δ means A is better.

## Outputs

- `evidence/bbob_noisy_d40_i1-15_switch_bootstrap_ci/pairwise_bootstrap_ci_switch_vs_cma_noisefree_B200.json`
- `evidence/bbob_noisy_d40_i1-15_switch_bootstrap_ci/pairwise_bootstrap_ci_switch_vs_cma_noisefree_B500.json`

## Reproduce

- `python3 tools/pairwise_bootstrap_ci.py --results-dir Results/bbob_noisy_d40_i1-15_switch_probe_t012_B200/noisefree --algo-a "Switch-MisrankingProbe(t=0.12)" --algo-b CMA-ES-sep --transform log10 --n-bootstrap 20000`
- `python3 tools/pairwise_bootstrap_ci.py --results-dir Results/bbob_noisy_d40_i1-15_switch_probe_t012_B500/noisefree --algo-a "Switch-MisrankingProbe(t=0.12)" --algo-b CMA-ES-sep --transform log10 --n-bootstrap 20000`
