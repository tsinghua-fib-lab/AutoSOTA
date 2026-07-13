# Robustness and Intervention Experiments

This folder contains the code and data for two robustness evaluations in the
ProtoT paper:

1. Prototype-Mediated Robustness (PMR): clamp ProtoT routing from an original
   input while evaluating a surface-form perturbation.
2. Intervention behavior: compare next-token distributions when a semantic tag
   changes the intended interpretation, such as gender, number, or negation.
   In this setting larger distribution shift is the expected sensitivity signal.

## Files

- [`perturbation_dataset/`](perturbation_dataset/): generation and filtering
  code, raw benchmark, cleaned benchmark, and dataset summary.
- [`intervention_dataset/intervention_benchmark_clean.jsonl`](intervention_dataset/intervention_benchmark_clean.jsonl): intervention
  benchmark used for the semantic sensitivity experiment.
- [`reproduce_amine.py`](reproduce_amine.py): command-line reproducer for the
  PMR and four-model intervention analyses.
- [`metrics.py`](metrics.py), [`models.py`](models.py), and [`tasks.py`](tasks.py):
  evaluation support code.

## Models

| Model | Implementation |
| --- | --- |
| ProtoT | [`../../prototype_attn.py`](../../prototype_attn.py) |
| Mamba | [`../../mamba.py`](../../mamba.py) |
| LLaMA | [`../../llama_baseline.py`](../../llama_baseline.py) |
| DeltaNet | [`../../deltanet.py`](../../deltanet.py) |

## Installation

Use Python 3.10 or newer. A minimal environment for these experiments is:

```bash
pip install torch transformers tokenizers pandas numpy scipy matplotlib seaborn datasets
```

See the [repository installation instructions](../../README.md) for the
additional DeltaNet dependencies.

## Checkpoints

Use the large FineWeb checkpoints. PMR requires ProtoT; the intervention
comparison requires all four models. Place the checkpoint files in this layout:

```text
experiments/robustness_clamping_and_intervention/
├── ProtoT/
│   ├── args.json
│   └── model_state_dict.pth
├── Mamba/
│   ├── args.json
│   └── model_state_dict.pth
├── LLaMA/
│   ├── args.json
│   └── model_state_dict.pth
└── DeltaNet/
    ├── args.json
    └── model_state_dict.pth
```

Each directory must contain the matching `args.json` and
`model_state_dict.pth` files.

## Validation data

Generate `data/FineWeb/val.npz` using the
[FineWeb data preparation instructions](../../fineweb_data_prep/README.md) before
running full perplexity recomputation.

## Reproduction

From the repository root:

```bash
python experiments/robustness_clamping_and_intervention/reproduce_amine.py --mode check-data
python experiments/robustness_clamping_and_intervention/reproduce_amine.py --mode all
```

Mode scope:

- `check-data`: validate both benchmarks; no checkpoints required.
- `pmr`: run prototype clamping with ProtoT only.
- `intervention`: run the gender, negation, and number comparison with ProtoT,
  Mamba, LLaMA, and DeltaNet.
- `all`: run PMR and the four-model intervention comparison.

For a quick smoke test with the checkpoint files in place:

```bash
python experiments/robustness_clamping_and_intervention/reproduce_amine.py --mode all --n-per-slice 2
```

Outputs are written to:

```text
experiments/robustness_clamping_and_intervention/results/amine_reproduction/
```

The key generated files are:

- `pmr_summary.csv`
- `pmr_call_table.csv`
- `intervention_summary_long.csv`
- `intervention_comparison_table.csv`

By default the script reads logged validation perplexities from
`final_val_ppl.txt`. Pass `--compute-full-ppl` only if you want to recompute
perplexity from `data/FineWeb/val.npz`, which is slower.
