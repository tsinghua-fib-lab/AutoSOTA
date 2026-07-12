# IOH Discrete Black-Box Optimization Benchmark

Example of true black-box optimization with OVLR:
- No gradients used, only objective function values
- On discrete PBO (Pseudo-Boolean Optimization) functions from IOHexperimenter
- Compared against standard black-box baselines (REINFORCE, CEM, (1+1)-EA)

## Default Problems

- `onemax`
- `leadingones`
- `linear`
- `isingring`
- `isingtorus`

Default dimension: 32, budget: 512 black-box evaluations.

## Methods

- `ovlr`: OVLR on a binary-search policy: optimize continuous logits `theta`, evaluate `sign(theta + noise)`
- `reinforce`: Independent Bernoulli policy gradient baseline
- `cem`: Cross-Entropy Method baseline
- `one_plus_one_ea`: Classic (1+1)-EA evolutionary algorithm baseline

All methods compared under the same **true function evaluation budget**.

## Install

```bash
pip install -r examples/black_box_optimization/requirements.txt
```

## Quick Start

Run all methods on default suite:

```bash
cd examples/black_box_optimization
python run_experiment.py --method all
```

Run only OVLR:

```bash
python run_experiment.py --method ovlr
```

Fast smoke test:

```bash
python run_experiment.py --method all --problems onemax,leadingones --budget 128 --seeds 0
```

## Outputs

Results written to `reported/discrete/`:
- `summary.csv` - Per-run summary
- `aggregate_problem.csv` - Per-problem aggregate
- `aggregate_overall.csv` - Overall method aggregate
- `report.md` - Human-readable markdown report
