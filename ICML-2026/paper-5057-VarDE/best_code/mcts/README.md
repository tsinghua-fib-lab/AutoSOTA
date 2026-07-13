# Monte Carlo Tree Search (MCTS)

C++ implementation and experiment utilities for MCTS-style algorithms.

## Build
From this folder, using a C++20-capable compiler (default: `g++`) and `make`:
- `make mcts-tune-frozenlake`
- `make mcts-tune-sailing`
- `make mcts-tune-taxi`
- `make mcts-tune-stree`
- `make mcts-run-frozenlake`
- `make mcts-run-sailing`
- `make mcts-run-taxi`
- `make mcts-run-stree`

## Outputs
- The Makefile `clean` target removes build artifacts under `mcts/bin/` and produced binaries.
- Plot artifacts typically go under `mcts/plots/` (excluded via `.gitignore`).

## Plotting
- `plot.ipynb` can be used to visualize results (notebook outputs are cleared for submission hygiene).

## Acknowledgements and license

Parts of the code in this folder are based on the THTS++ project by MWPainter (branch `xpr_go`): https://github.com/MWPainter/thts-plus-plus/tree/xpr_go. The upstream project is licensed under the GNU GPLv3 (see upstream `LICENSE`).

If you redistribute or publish derivative code that includes substantial portions of the upstream project, ensure you comply with the GPL-3.0 requirements (e.g., provide source and include the GPL notice). See the upstream repository for full license and attribution details.

Suggested citation for the upstream work:

MWPainter. "Monte Carlo Tree Search with Boltmann Exploration." (see https://github.com/MWPainter/thts-plus-plus/tree/xpr_go and https://arxiv.org/abs/2404.07732)

