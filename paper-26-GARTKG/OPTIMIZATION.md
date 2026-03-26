# Paper 26 — GARTKG

**Full title:** *A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning*

**Registered metric movement (internal ledger, ASCII only):** +12.4%(49.44->55.59)

# Final Optimization Report: GARTKG (paper-26)

## Summary

Primary **temporal KG reasoning** accuracy improved from **49.44% → 55.59%** on the **final** checkpoint with **replay weight 0.3** on the **`_best`** export. An earlier iteration reportedly reached **~64.42%** but a **same-name checkpoint** was overwritten by later iterations—see internal **scores** notes for lineage.

## Key ideas (results ledger)

- Tune **generative adaptive replay** weight (final **0.3** on **`_best`**) for stability vs plasticity on evolving graphs.

## Caveats

- **Metric vs checkpoint naming:** when every iteration writes the same filename, only the **last** export is guaranteed in `optimized_code`.

## Where to look next

- **`README.md`** and training config for replay weight and metric logging.
