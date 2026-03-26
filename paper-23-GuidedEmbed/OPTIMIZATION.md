# Paper 23 — GuidedEmbed

**Full title:** *Don’t Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation*

**Registered metric movement (internal ledger, ASCII only):** +17.5%(36.05->42.37)

# Final Optimization Report: Guided space transformation (paper-3)

## Summary

After **skipping full repro**, the SOTA loop focused on training and evaluation hyperparameters for the guided embedding objective. **v_measure** (primary) moved from **36.05 → 42.37 (+17.5%)**, with the strongest checkpoint appearing around **iteration 12** in the optimizer trace.

## Key ideas (results ledger)

- Multi-round sweeps over **learning rate schedules**, **batch construction**, and **projection / transformation** knobs that control how instruction pairs are aligned in the shared space.

## Context

- Because repro was skipped, treat numbers as **internal-harness** results tied to this snapshot; re-run the published eval script if you need paper-identical settings.

## Where to look next

- **`README.md`** for data preparation; config YAML or trainer flags checked in by the winning iteration.
