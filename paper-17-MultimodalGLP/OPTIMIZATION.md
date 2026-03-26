# Paper 17 — MultimodalGLP

**Full title:** *Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection*

**Registered metric movement (internal ledger, ASCII only):** +0.83%(0.8763->0.8836)

# Final Optimization Report: Global label propagation (paper-17)

## Summary

Closed-loop optimization raised **accuracy** from **0.8763 → 0.8836 (+0.83%)** by giving the graph-based label propagation stage more iterations to converge before evaluation.

## Key change (results ledger)

- Set **`n_label_iters` from 1 → 3** on the train/test path so each batch benefits from deeper global propagation rounds instead of a single shallow sweep.

## Context

- Repro was **skipped** for this run per pipeline policy; SOTA used the shipped optimized codebase and the internal evaluation harness.
- Multi-seed checks showed an `n_label_iters=1` variant that **hurt** accuracy; that setting was rolled back before locking the deeper schedule.

## Where to look next

- See the project **`README.md`** for dataset paths, default configs, and the exact script entrypoints used in eval.
