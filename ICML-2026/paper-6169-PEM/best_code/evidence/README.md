# Evidence (precomputed artifacts)

This folder contains precomputed artifacts (CSVs/JSON/plots/PDFs) for the experiments.

Conventions:
- Each subfolder corresponds to one experiment family or one figure/table group.
- Each subfolder includes a short `README.md` describing the setup and key files.
- `docs/FIGURES.md` maps major claims/figures to the most important files under `evidence/`.

Regenerating:
- To rerun experiments end-to-end (writes `Results/` / `exdata/` and refreshes `evidence/`):
  `python3 tools/reproduce_all.py --workers 4`
- To regenerate derived plots/tables from existing raw outputs (if available):
  `python3 tools/refresh_artifacts.py --workers 4`
