# Geometry

## Quick Start

```bash
# Reproduce the ant parameterization result
python mapping.py parameterization 0.1
```

## Arguments

```
python mapping.py [example] [lambda_reg]
```

| Argument         | Description                             |
| ---------------- | --------------------------------------- |
| `[example]`    | `parameterization`.                   |
| `[lambda_reg]` | Regularization parameter (e.g.`0.1`). |

There are additional settings in `mapping.py`. The QP is solved within `mapping_layer.py`, where the solver and other options can be modified. Results are saved to the `results/` directory.
