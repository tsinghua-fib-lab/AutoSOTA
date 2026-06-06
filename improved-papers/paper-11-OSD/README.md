# Paper 11 — OSD

**Full title:** *Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection*

**Original codebase:** This optimization is based on the [*Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection*](https://github.com/YZY-stack/Effort-AIGI-Detection) repository. For the original paper, see [arXiv:2411.15633](https://arxiv.org/abs/2411.15633).

**Registered metric movement (internal ledger):** +0.25%(99.35→99.6)

## Summary

Final BigGAN accuracy increased from **99.35** to **99.6** through inference-time refinements rather than architecture changes: **horizontal-flip TTA**, **L2 feature normalization**, and LEAP-style feature construction (`CLS + patch mean`). The best setting used **asymmetric 3-view TTA weights** `orig/flip/cls-only = 0.7/0.2/0.1`.

## Key ideas

- Keep the base detector fixed and optimize **test-time views/weights**.
- Normalize features before fusion to reduce view-magnitude bias.
- Lightweight TTA weighting can still move metrics near the saturation zone.

## Where to look

- `UniversalFakeDetect_Benchmark/` inference and model wiring code
- model feature extraction and aggregation paths used in eval
