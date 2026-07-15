# tslviz

Browser-based visualizer for [TSL](https://github.com/jyliuu/TSL) split-event
logs. TSL (Tensor Separation Learning) is a glass-box regression model that
fits a sum of *stages*. Each stage is the ordered difference of two
non-negative rank-1 *products*, and each product is a product of non-negative
one-variable *components* — one per feature. Each component pair on a feature
also admits a backbone/tilt view: the *backbone* carries the shared magnitude
and gates where the stage is active, the *tilt* carries the signed imbalance
between the two products and determines the stage's direction.

`tslviz` lets you step through every split, resplit, and merge that produced
these products; compare combined per-stage predictors across runs; and
inspect the components, their backbone/tilt decomposition, and the per-product
scalings that define each stage.

The backend is FastAPI; the frontend is static HTML + D3.js. Both ship together
in a single `pip install`.

## Installation

```bash
pip install tslviz
```

For development:

```bash
git clone https://github.com/jyliuu/TSL
cd TSL/tsl-split-evolution-dashboard
pip install -e .
```

## Running

Point `tslviz` at the SQLite database written by a TSL fit
(produced when `visualdb` is set on the fit parameters):

```bash
tslviz --db path/to/your.sqlite
tslviz --db path/to/your.sqlite --port 8080 --reload
```

Or via the environment variable:

```bash
export DATABASE_PATH=path/to/your.sqlite
tslviz
```

Open <http://localhost:8051/> in a browser.

## Pages

| Page                    | What it shows                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Run summary**         | Per-run learning curves, energy, scaling, column-level split statistics, convergence.                                    |
| **Product evolution**   | Every bagged positive and negative rank-1 product across fit iterations; overlay the aggregated stage predictor.         |
| **Combined products**   | Per-stage aggregated predictor (positive product minus negative product) overlaid across selected runs.                  |
| **Component evolution** | The one-dimensional components on each feature axis across stages, with optional backbone / tilt view.                   |
| **Backbone / tilt**     | Per-axis backbone (shared magnitude of the positive/negative pair) and tilt (their signed imbalance).                    |
| **f+ / f-**             | The two non-negative branch components — the positive and negative one-dimensional factors — rendered side by side.      |
| **Lambda scatter**      | Per-product positive vs. negative stage scalings, for spotting outlier products within a stage.                          |

## Generating a database from TSL

In Python, pass `visualdb_path` to the fit; the dashboard reads from the same
file:

```python
from tensorsl.sklearn import TSLRegressor

model = TSLRegressor(
    epochs=5, n_trees=20, n_iter=120,
    visualdb="run.sqlite",
)
model.fit(X, y)
```

Then in a separate shell:

```bash
tslviz --db run.sqlite
```

## Notes

- Read-optimized SQLite indexes are created at startup; the dashboard never
  writes to the database.
- The frontend is served as static files from `/`.

## License

MIT.
