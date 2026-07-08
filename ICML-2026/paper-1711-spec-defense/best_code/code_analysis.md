# CSR Code Analysis for SOTA Optimization

## Evaluation Path
- Entry: run_reproduction.sh -> scripts/evaluate.py
- Adversarial samples pre-generated: scripts/generate_adv.py (PGD, eps=1/255, steps=10)
- Defense: FastCSRDefense in csr/defense/csr_fast.py
- Evaluation: CLIPBenchmark in csr/evaluation/benchmark.py

## Key Files
| File | Role | Safe to Modify |
|------|------|---------------|
| csr/defense/csr_fast.py | Fast CSR defense (detect + purify) | YES - core defense logic |
| csr/config.py | CSRConfig dataclass | YES - add config fields |
| configs/default.yaml | Default hyperparameters | YES - parameter tuning |
| scripts/evaluate.py | Eval entry point (CLI args) | YES - expose new CLI args |
| run_reproduction.sh | Top-level eval script | YES - pass new params |
| csr/evaluation/benchmark.py | Benchmark runner | CAUTION - eval protocol |
| csr/evaluation/evaluator.py | Zero-shot evaluator | CAUTION - metric computation |
| scripts/generate_adv.py | Adversarial sample generator | NO - changes eval protocol |

## Current CSR Hyperparameters
- filter_type: gaussian
- lpf_radius: 40
- butterworth_order: 2 (not exposed via CLI)
- detect_thresh: 0.85
- purify_steps: 3
- purify_eps: 4/255 (not exposed via CLI)
- purify_alpha: 2/255 (not exposed via CLI)

## CLI Args Currently Exposed
- --lpf_radius, --detect_thresh, --purify_steps, --filter_type
- NOT exposed: --purify_eps, --purify_alpha, --butterworth_order

## Red Lines
- Do NOT modify: metric computation, data splits, labels, scoring
- Do NOT hard-code: predictions, metric values, dataset-specific shortcuts
- Do NOT change: eval protocol (dataset, attack, sample count)
