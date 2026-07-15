# Visualization Dashboard (WIP)

The `tslviz` dashboard replays how a TSL model was built — every split, the
backbone/tilt evolution, the $\tilde{m}_+/\tilde{m}_-$ branches, and the per-stage aggregation.
It reads a SQLite database that the core writes during a fit (the
[evo-logging](../code/logging.md) system).

`tsl-split-evolution-dashboard/` is a standalone **FastAPI + D3** app, packaged as
`tslviz`. It is independent of the core: you fit with `visualdb=...` set, then point
`tslviz` at the resulting `.sqlite` file.

## 1. Fit with a `visualdb` path

Set `visualdb` to a SQLite file when fitting (requires the `evo-logging` feature, on by
default):

```python
from tensorsl.sklearn import TSLRegressor

model = TSLRegressor(epochs=5, n_trees=16, n_iter=30, seed=0, visualdb="run.sqlite")
model.fit(X, y)
```

Every split / resplit / merge, component snapshot, and per-stage scaling is streamed to
`run.sqlite`.

## 2. Launch the dashboard

Install and run `tslviz` (a separate package, `tsl-split-evolution-dashboard/`):

```bash
pip install ./tsl-split-evolution-dashboard   # or from the repo
tslviz --db run.sqlite                          # serves http://localhost:8051
tslviz --db run.sqlite --port 8080 --reload     # custom port with autoreload
```

Alternatively, set the `DATABASE_PATH` environment variable and run `tslviz` with no
`--db` flag, or invoke directly via `uvicorn tslviz.backend.app:app`. The entry point is
`tslviz.backend.app:main` (`pyproject.toml`). Dependencies are `fastapi`,
`uvicorn[standard]`, and `orjson`.

## 3. Explore

The single-page UI is backed by the read-only [JSON API](#the-json-api). Typical things
to look at:

- **Timeline / learning curves** — how error dropped split by split and epoch by epoch.
- **Tree evolution** — reconstruct a single grid's interval structure up to any iteration.
- **Backbone / tilt evolution** — watch the magnitude gate and signed direction form per
  feature.
- **$\tilde{m}_+ / \tilde{m}_-$ components** — the two positive branches whose difference is the stage.
- **λ scatter / scalings / energy** — diagnose aggregation across the bagged grids
  (the [bimodal-alignment](../math/bagging-aggregation.md#the-bimodal-alignment-example)
  issue shows up here).

The database is opened read-only, so the dashboard never modifies your run.

## Layout

| Path | Role |
|------|------|
| `backend/app.py` | the entire FastAPI app: DB connection, JSON API, static-file serving |
| `frontend/index.html`, `styles.css` | the single-page D3 front end |
| `pyproject.toml` | packaging + the `tslviz` console script |

On startup the backend opens the DB with read-optimized pragmas (WAL, large cache) and
creates helper indexes; responses use `orjson` (encoding non-finite floats as `null`), GZip,
and permissive CORS.

## What it reads

The backend queries the tables the core's logger writes (see [Logging](../code/logging.md)):

- **`runs`** — run metadata (rows, cols, params).
- **`events`** — split / resplit / merge actions (epoch, tree, iteration, column, split
  value, gain, counts).
- **`component_states`** — per-iteration backbone/tilt per feature (binary blobs).
- **`combined_grids`** — per-stage aggregated grid snapshots.
- **`epoch_scalings`**, **`training_errors`**, **`combination_choices`**,
  **`f_component_stats`**, **`tensor_lambdas`** — scaling, learning curves, aggregation
  choices, and $\tilde{m}_+/\tilde{m}_-$ branch statistics.

## The JSON API

`backend/app.py` exposes a read-only API consumed by the front end; a representative slice:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/runs` | list runs with summary counts |
| `GET /api/run/{id}/timeline` | ordered split events |
| `GET /api/run/{id}/learning`, `/convergence` | per-epoch learning / convergence curves |
| `GET /api/run/{id}/tree_evolution?epoch&tree_id&iteration` | reconstruct a tree's state up to an iteration |
| `GET /api/run/{id}/backbone_tilt_evolution[_all_columns]` | backbone/tilt over iterations |
| `GET /api/run/{id}/f_component_evolution`, `/f_component_per_axis` | $\tilde{m}_+/\tilde{m}_-$ branch evolution |
| `GET /api/run/{id}/identified_components[_all]` | final per-feature components after identification |
| `GET /api/run/{id}/tensor_lambdas`, `/scalings`, `/energy` | $\lambda_\pm$, OLS scalings, stage energy |

Helper routines reconstruct tree geometry by replaying split/resplit/merge events and decode
the binary `f64` component blobs (the same `encode_f64_array`/`decode_f64_array` convention
used by the core). The logging output this dashboard consumes is exercised by
`tests/evo_logging.rs` in the core crate.
