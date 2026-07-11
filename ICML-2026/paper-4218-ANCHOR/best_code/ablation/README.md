# ANCHOR — Component Ablation (Claude Code Haiku 4.5)

Ablation study isolating the contribution of each ANCHOR auditor-agent component.
The full auditor drives **Claude Code (Haiku 4.5)** as the target; each variant removes
one component and re-runs the same 10 cases (indices 000–009, `MAX_TURNS=100`).

| Condition | Script | Removed | Result |
|---|---|---|---|
| Baseline | `auditor_baseline.py` | nothing | 10/10 success, avg 8.3 turns |
| Vanilla Model | `auditor_ablation_vanilla_model.py` | SFT+RL trained auditor weights | 1/10 — base model refuses to plan |
| No Toolbox | `auditor_ablation_no_toolbox.py` | strategy options A–E | 2/10 — pipeline loses todo, fragile |
| No Rewrite | `auditor_ablation_no_icrl_norewrite.py` | ICRL + best-of-N rewriting | 10/10 but 15.8 turns (efficiency only) |

**Takeaway:** the trained model and the strategy toolbox are both necessary; the rewriting
pipeline is an efficiency multiplier, not a requirement for success. Full write-up:
[`ABLATION_REPORT.md`](ABLATION_REPORT.md). Paper table source: [`ablation_table.tex`](ablation_table.tex).

## Layout

```
ablation/
├── README.md
├── ABLATION_REPORT.md
├── ablation_table.tex
└── scripts/
    ├── auditor_baseline.py
    ├── auditor_ablation_vanilla_model.py
    ├── auditor_ablation_no_toolbox.py
    ├── auditor_ablation_no_icrl_norewrite.py
    ├── response_pipeline.py            # shared response-parsing helpers
    └── xml_response_formatter.py       # shared XML formatting helpers
```

Task data is read from the shared `../sample_data/second_phase_instructions/sampled_300_first_iteration_3_judges.json` (ANCHOR-Seed task set).

## Requirements

- `openai` Python package (used as the OpenRouter client) and the `claude` CLI (Claude Code) on PATH.
- The Thinking Machines `tinker` SDK to serve the trained auditor checkpoints.

Set these placeholders before running:
- In each script, `TINKER_MODEL_PATH` / `REWRITE_MODEL_PATH` use
  `YOUR_SFT_CHECKPOINT_UUID` / `YOUR_RL_CHECKPOINT_UUID` — replace with your own
  Tinker checkpoints (the trained ANCHOR auditor is required; the scripts do not run
  end-to-end on the base model except for the vanilla-model ablation).
- Environment variables: `ANTHROPIC_API_KEY` (Claude Code target), `OPENROUTER_API_KEY`
  (vanilla-Qwen evaluator), `TINKER_API_KEY` (auditor checkpoints).

## Run

```bash
export ANTHROPIC_API_KEY=...   # Claude Code Haiku 4.5 target
export OPENROUTER_API_KEY=...  # vanilla Qwen3-235B evaluator
export TINKER_API_KEY=...      # trained auditor checkpoints
export NUM_SAMPLES=10 MAX_TURNS=100

python scripts/auditor_baseline.py                    # → ./results_baseline/
python scripts/auditor_ablation_vanilla_model.py      # → ./results_vanilla_model/
python scripts/auditor_ablation_no_toolbox.py         # → ./results_no_toolbox/
python scripts/auditor_ablation_no_icrl_norewrite.py  # → ./results_no_icrl_norewrite/
```

Override the output location with `BASE_DIR`. Score the resulting trajectories with the
catastrophic judge in [`../evaluation_judges/`](../evaluation_judges/).
