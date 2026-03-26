# Paper 21 — TokenRecycling

**Full title:** *Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling*

**Registered metric movement (internal ledger, ASCII only):** +19.5%(2.7386->3.2726)

# Final Optimization Report: Token recycling (paper-21)

## Summary

Speculative decoding with token recycling was tuned over multiple iterations (tree / MAT-style configurations and acceptance policy). **Mean accepted tokens** improved from **2.7386 → 3.2726 (~+19.5%)** versus the baseline run; the best score in logs landed around **iteration 10**.

## Key ideas (results ledger)

- Joint tuning of **tree / MAT** draft geometry and **token recycling** behavior so more draft tokens survive verification without blowing up rejection rates.

## Context

- Formal repro was **skipped**; optimization ran on the frozen optimized snapshot.
- Later iterations that moved away from the winning draft geometry were **rolled back** when `mean_accepted_tokens` dropped.

## Where to look next

- Inspect optimizer **`scores.jsonl`** (if present) for per-iter acceptance stats; upstream **`README.md`** documents baseline launch flags.
