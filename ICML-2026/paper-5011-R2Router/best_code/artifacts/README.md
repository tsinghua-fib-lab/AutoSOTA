# Artifact Scope

The public artifact should include only the research code and reproducibility metadata needed for:

- R2-Bench / IID experiments in `main/`
- RouterArena routing experiments in `scripts/`
- optional OOD evaluation in `ood_evaluation/`

The following are intentionally treated as local-only and should be excluded from the public release package:

- `demo/`
- `old_demo/`
- `hf_space/`
- `hf_upload/`
- cached checkpoints
- submission JSONs containing generated outputs
- local logs, plots, and temporary tarballs

Suggested release workflow:

1. Keep this repo as the working repository.
2. Create a separate release branch or public mirror.
3. Remove local-only directories there.
4. Upload large artifacts separately and document download steps in `README.md`.
