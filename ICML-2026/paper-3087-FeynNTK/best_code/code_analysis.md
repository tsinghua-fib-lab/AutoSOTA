# Code Analysis for Paper 3087

## Evaluation Path

- Main command: `JAX_PLATFORMS=cpu uv run recursions recursion-width-study H1 --n_layers 4 --layer_widths 20 --C_w 1.98305826 --input_file examples/inputs/input_recursions.json --input_rescale 0.53 --max_subdiv 10000 --rtol 1e-3 --output_dir eval_output`
- Entry: `src/ntkunlimited/recursions/recursions.py:cli()` -> `recursion_width_study` command
- Target tensor: `H1` (requires V4, G1, F, D as dependencies)
- Results: JSON files in `eval_output/tensors/`

## Metric Parsing

- NNGP_K01_corrected = G["0-1"] + G1["0-1"]/n
  - G: from `G_analytic_*.json` -> `data["width-20"]["layer-4"]["0-1"]`
  - G1: from `G1_analytic_*.json` -> `data["width-20"]["layer-4"]["0-1"]`
  - n = layer_width = 20

- NTK_Theta01_corrected = H["0-1"] + H1["0-1"]/n
  - H: from `H_analytic_*.json` -> `data["width-20"]["layer-4"]["0-1"]`
  - H1: from `H1_analytic_*.json` -> `data["width-20"]["layer-4"]["0-1"]`

## Key Files

| File | Role |
|---|---|
| `recursions.py:800-1204` | CLI entry, recursion-width-study command |
| `recursions.py:187-266` | comp_G1s (finite-width NNGP correction) |
| `recursions.py:432-550` | comp_H1s (finite-width NTK correction) |
| `numerical_integration.py:1-91` | CubatureConf, normal_expec, integration |
| `symbolic_to_numerics.py` | GaussExpecNumeric (caching + numeric eval) |
| `config.py` | Cache directory |
| `recursion_symbols.py` | Symbolic kernel definitions |

## Integration Method

- `normal_expec()` uses scipy cubature with:
  - `genz-malik` rule for dim > 1
  - `gauss-kronrod` rule for dim = 1
- Controlled by `CubatureConf(rtol, max_subdiv)`

## Caching

- Gaussian expectation results cached in `src/ntkunlimited/recursions/cache/gaussexpec_numeric_cache.pkl`
- Keyed by CubatureConf + integrand hash + covariance
- Must clear cache when changing rtol/max_subdiv (or use --force-recompute)
- Recursion numeric functions also cached as .pkl files

## Safe Modification Targets

1. `numerical_integration.py:25-30` — CubatureConf and integration rule selection
2. `recursions.py:847-851` — CLI defaults for rtol, max_subdiv
3. `recursions.py:821` — C_b parameter (currently None)
4. `recursions.py:868` — input_rescale parameter
5. `recursions.py:815` — C_w parameter

## Risky Files (avoid modifying)

- `recursion_symbols.py` — defines the actual symbolic recursion
- `G1_expressions.py`, `H1_expressions.py`, etc. — derived from paper formulas
- Test data, input files
