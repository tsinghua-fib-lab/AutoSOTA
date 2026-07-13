# Final Release Checklist

Use this checklist when preparing the public RouterArena release branch.

## 1. Clean The Public Branch

Remove local-only tracked content:

- `demo/`
- `routerarena_submission/*.json`
- `ood_evaluation/demo_plots/`
- `CLAUDE.md`

Consider also removing from the public branch if not needed:

- `data_collection/`
- `main/`
- `R2-Router.pdf`
- cluster-specific `scripts/*.sbatch`

## 2. Keep The RouterArena Core

Ensure these remain in the public code repo:

- `scripts/`
- `reproduce/`
- `artifacts/`
- `.env.example`
- `README.md`
- `pyproject.toml`

## 3. Upload The Data Package

Publish a RouterArena data package containing:

- `budget_sweep/`
- `category_router/training_data.pkl`
- `embeddings/routerarena_embeddings.pkl`
- `embeddings/routerarena_robustness_embeddings.pkl`
- `routerarena_meta/router_data.json`
- `routerarena_meta/router_data_10.json`
- `routerarena_meta/model_cost.json`

Include:

- `README.md`
- `LICENSE_NOTE.md`

Suggested base template:

- `artifacts/routerarena_data_release_README.md`
- `artifacts/LICENSE_NOTE.md`

## 4. Upload Optional Checkpoints

If publishing pretrained checkpoints, release:

- `checkpoints/category_router`

Do not release:

- `checkpoints/category_router_mixed_backup`

Use:

- `artifacts/checkpoints_release_README.md`

## 5. Verify Reproduction Path

Test the public path with:

```bash
bash reproduce/routerarena_train.sh
bash reproduce/routerarena_eval.sh
```

Confirm that all required environment variables are documented and that no private absolute paths remain on the critical path.

## 6. Final Metadata

Before publishing, make sure you have:

- repository description
- version tag or release name
- citation text
- contact / takedown line
- dependency version note
- upload destination for large artifacts

## 7. Final Sanity Check

Run:

```bash
git ls-files
git status --short
```

Make sure the public branch does not contain:

- certificates
- pid files
- logs
- local plots
- submission outputs
- unused backups
