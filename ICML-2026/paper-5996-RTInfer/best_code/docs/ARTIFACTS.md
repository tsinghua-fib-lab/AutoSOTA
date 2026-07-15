# Artifact Structure

## Source Code

- `rtinfer/model.py`: model, variant, task, and job data structures.
- `rtinfer/atlas.py`: pruning/early-exit variant atlas and Pareto filtering.
- `rtinfer/layout.py`: time-address memory layout solver.
- `rtinfer/delta_graph.py`: chunking, residency, and load-time model.
- `rtinfer/scheduler.py`: online policies, baselines, and ablations.
- `rtinfer/pantheon_io.py`: Pantheon profile/workload import.
- `rtinfer/simulate.py`: command-line simulator.

## Experiments

- `rebuttal_experiments/`: modern/reviewer-aligned deterministic experiments.
- `case_studies/jetson_nano_case/`: explanatory figure and traces.
- `scripts/`: wrappers for Pantheon-profile simulations.
- `tools/`: Jetson profiling helpers.

## Generated Outputs

Generated outputs should not be committed by default.

- `outputs/`: experiment logs and CSV summaries.
- `paper_figures/revised_outputs/`: regenerated paper-style SVG/PDF figures.
- `case_studies/jetson_nano_case/outputs/`: deterministic case-study traces.
- `tmp_pdf/`: local PDF extraction scratch space.

## External Inputs

The public repository does not include private datasets, model weights, or
Pantheon artifacts. Users can provide them through:

- `--pantheon-repo`
- `--profile-root`
- `--deploy-json`
- `--workload-json`

## Reproducibility Notes

- The core simulator is deterministic.
- Synthetic modern profiles are calibrated assumptions, not measured accuracy.
- Hardware profiling scripts may vary across Jetson software images, clock
  modes, and thermal states.
