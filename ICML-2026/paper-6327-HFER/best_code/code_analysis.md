# Code Analysis - Paper 6327 SOTA Optimization

## Evaluation Path
- Eval command: python3 scripts/analysis/recalc_full_stats.py
- Input: data/results/experiment_results_Qwen2.5-0.5B-Instruct.json
- Label correction: data/reclaimed/Qwen0.5B_list_b_confident_invalid.json
- Output: ranked table of (layer, metric, p-value, Cohen d, Accuracy)
- Baseline: Accuracy=93.2%, Cohen d=2.93 at entropy Layer 0

## Data Flow
1. run_experiment.py loads Qwen2.5-0.5B-Instruct, passes proofs through model
2. spectral_trust library constructs attention graphs, computes Laplacian, extracts spectral features
3. Features stored per proof: fiedler_value, energy, smoothness, entropy, hfer at each of 24 layers
4. recalc_full_stats.py computes Mann-Whitney U, Cohen d, accuracy via optimal threshold
5. 454 proofs: 154 valid + 300 invalid -> 194 valid + 260 invalid after label correction

## Key Files
- scripts/analysis/recalc_full_stats.py - EVALUATION SCRIPT (analysis-only)
- scripts/run_experiment.py - Feature extraction (45 min on A100)
- spectral_trust/config.py - GSPConfig dataclass
- spectral_trust/graph.py - GraphConstructor
- spectral_trust/spectral.py - SpectralAnalyzer
- spectral_trust/framework.py - GSPDiagnosticsFramework pipeline
- scripts/analysis/find_perfect_combo.py - AND-rule reference

## Safe Modification Targets
- recalc_full_stats.py - analysis-only changes
- spectral_trust/spectral.py - new metrics (alpha, band energies)
- spectral_trust/graph.py - graph construction changes
- spectral_trust/config.py - new parameters
- run_experiment.py - extract new metrics

## Risky Files (do NOT modify)
- data/results/experiment_results_Qwen2.5-0.5B-Instruct.json
- data/reclaimed/*.json
- data/experiment_ready/
