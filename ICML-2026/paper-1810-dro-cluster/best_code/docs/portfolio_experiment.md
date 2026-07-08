# Financial Portfolio Experiment

The manuscript portfolio experiment used licensed WRDS/CRSP-Compustat data and a local `portfolio_backtesting` workflow.
Those raw files and large pickle intermediates are intentionally excluded from this GitHub release.

## Required Local Data

The original scripts expect daily S&P 500 data with these files:

- `daily_adj_close.csv`: adjusted close prices with a `datadate` column and one column per security id.
- `daily_close.csv`: unadjusted close prices with the same date/security layout.
- `shares_outstanding.csv`: shares outstanding with the same date/security layout.
- `all_constituents.csv`: S&P 500 membership intervals, including `gvkey-iid`, `from`, and `thru`.
- SPY benchmark data: the original code queried WRDS `comp.idx_daily` for the configured index key.

## Source Provenance

The compact release does not repackage the full backtester as a standalone script because the manuscript run depends on
licensed data, old local package state, and large intermediates:

- `code/subspace_clustering/portfolio.py` and `portfolio_analysis.ipynb` define the experiment configuration.
- `code/portfolio-backtesting/portfolio_backtesting/portfolio.py` contains the rolling backtest shell.
- `code/portfolio-backtesting/portfolio_backtesting/stock_selection/selection.py` contains the low-volatility
  representative stock selection rule used by the manuscript configuration.
- The archived `portfolio_backtesting.stock_selection.clustering` source contains the ACC/CORD,
  Lasso nodewise-regression, square-root Lasso, k-medoids, and hierarchical clustering helpers used by the
  portfolio code. The compact release ports the algorithmic helpers needed by the reproduction scripts.

For that reason, this release documents the financial experiment and preserves the available source-grounded DRO/CORD
building blocks, but does not include a substitute portfolio backtest script.
