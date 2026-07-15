# Logging (evo-logging)

`src/logging.rs` and `src/logging/` implement **evo-logging**: when a fit has a
`visualdb_path` set (and the `evo-logging` feature is on — it is by default), every
split / resplit / merge and per-stage snapshot is streamed to a SQLite database. The
[Visualization dashboard](../guides/visualizing.md) later reads that DB read-only to replay how each stage
was built.

When the feature is **off**, all the logging entry points compile to no-ops (the `pub fn`
stubs in `logging.rs`), so the core has zero logging overhead.

## Module layout

| File | Role | Feature-gated |
|------|------|:-:|
| `types.rs` | event/snapshot type definitions, `LoggingConfig` | no |
| `context.rs` | thread-local epoch / tree-id context (correct attribution under parallel fits) | no |
| `reducer.rs` | logging reducers that capture events from state changes | yes |
| `evo.rs` | logger trait + implementations (no-op and SQLite) | — |
| `manager.rs` | logger lifecycle / dispatch | — |
| `buffers.rs` | event buffering before write | yes |
| `sqlite_logger.rs` | the SQLite backend (buffered writes, schema) | yes |
| `simple.rs` | a minimal logger variant | — |

## What gets logged (`types.rs`)

- **`Action`** — `Split` / `Resplit` / `Merge`.
- **`SplitEvent`** — a split/resplit/merge: column, split value, error reduction, left/right
  counts, optional per-point residual `Update`s.
- **`CombinedGridSnapshot`** — a per-stage snapshot: epoch, energy, scaling, grid JSON, and
  $\tilde{m}_+/\tilde{m}_-$ arrays.
- **`EpochScalingSnapshot`** — the scaling coefficients each time they are re-optimized.
- **`ComponentStateSnapshot`** — per-feature component state (backbone/tilt per interval, in
  binary form) at an iteration.
- **`FComponentStats`** — summary statistics of an $\tilde{m}_+$ or $\tilde{m}_-$ branch.
- **`LoggingConfig`** — DB path, run label, and flags (`record_residual_updates`,
  `pack_updates_as_blob`).

## How events are captured

The grid-tensor reducer calls into the logging reducers (`logging/reducer.rs`,
`src/grid_tensor/logging_helpers.rs`) as it applies each action: split events, component
snapshots, and $\tilde{m}_\pm$ branch statistics are buffered with their epoch/tree context
(`logging/context.rs`) and flushed to SQLite. Binary arrays are encoded with the
`encode_f64_array` / `decode_f64_array` helpers shared with the dashboard backend.

## Using it

Pass a database path at fit time — from Python:

```python
TSLRegressor(..., visualdb="run.sqlite").fit(X, y)
```

then explore it with `tslviz --db run.sqlite`. The end-to-end workflow and the DB schema
the dashboard reads are documented on the [Visualization dashboard](../guides/visualizing.md)
page. The logging output is exercised in `tests/evo_logging.rs`.
