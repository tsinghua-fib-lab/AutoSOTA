# Synthetic Experiments

The synthetic sanity check compares the unbounded `(C,G)` certificate against alpha-smoothing on three 2D functions: quadratic, slice, and sandwich.

## Smoke Test

```bash
python experiments/synthetic/run_unbounded_synthetic.py \
  --function unbounded_quadratic \
  --sigma 0.1 \
  --eps_y 0.2 \
  --N_samples 200 \
  --n_test_points 2 \
  --alpha_trim 0.49 \
  --P 0.9 \
  --alpha_n_tr 200 \
  --alpha_n_sample 200 \
  --alpha_cp_alpha 0.001 \
  --skip_true_radius \
  --output outputs/smoke_synthetic.json
```

## Camera-Ready Grid

Run each `(function, eps_y, sigma, alpha)` combination and then summarize. The camera-ready grid uses:

- functions: `quadratic`, `slice`, `sandwich`
- `eps_y in {0.2, 0.5}`
- `sigma in {0.1, 0.2, 0.5}`
- `alpha_trim in {0.35, 0.49}`
- `N_samples=5000`, `alpha_n_tr=5000`, `P=0.9`, `alpha_cp_alpha=0.001`

Example single full run:

```bash
python experiments/synthetic/run_unbounded_synthetic.py \
  --function all_unbounded \
  --sigma 0.5 \
  --eps_y 0.5 \
  --N_samples 5000 \
  --n_test_points 10 \
  --alpha_trim 0.49 \
  --P 0.9 \
  --alpha_n_tr 5000 \
  --alpha_n_sample 200 \
  --alpha_cp_alpha 0.001 \
  --compute_true_radius \
  --output outputs/synthetic_sigma0p5_eps0p5_alpha0p49.json
```
