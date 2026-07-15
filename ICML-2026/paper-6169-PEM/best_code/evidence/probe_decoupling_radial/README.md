# Probe Decoupling Evidence (Radial / State-Dependent Noise)

This folder contains a minimal, review-friendly snapshot showing that a pointwise
variance proxy at `x0` can fail to detect *misranking*, while a candidate-set probe
captures it.

Experiment:
- Suite: `bbob-largescale` wrapped with **state-dependent** noise
- Noise model: `radial_additive_rel` (noise scale grows with RMS normalized distance from `initial_solution`)
- Noise sigma: `0.5`
- Dims: `80,160,320`
- Functions: `1,2,6,10,15,20`
- Instances: `1–15`
- Budget: `200xD`

Files:
- `probe_values.csv`: per-problem probe values and trigger flags (read trigger counts from this CSV).
- `probe_decoupling.png`: visualization (misranking RD varies while variance proxy stays ~0; trigger counts).
