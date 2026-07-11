# UTKFace Age-Estimation Experiments

This experiment certifies a pretrained MiVOLO-v2 age regressor on UTKFace. It compares `(E,C)+M`, `(E,C,G)+M`, and alpha-smoothing.

## Inputs

- UTKFace directory, passed with `--utk_dir`.
- MiVOLO-v2 Hugging Face checkpoint directory, passed with `--model_dir`.

## Smoke Test

```bash
python experiments/utkface_age/utkface_bounded_vs_alpha_experiment.py \
  --utk_dir data/UTKFace/UTKface_inthewild \
  --model_dir models/mivolo_v2_hf \
  --out_json outputs/utkface_smoke.json \
  --n_points 2 \
  --sigma 0.25 \
  --alpha 0.35 \
  --N 128 \
  --alpha_n_tr 128 \
  --alpha_n_sample 64 \
  --batch_size 64 \
  --mode both
```

## Full Grid

Camera-ready grid:

- `sigma in {0.06,0.12,0.25,0.5,0.75}`
- `alpha in {0.35,0.49}` for alpha-smoothing
- `n_points=100`
- `N=5000` for our methods
- `alpha_n_tr=5000`
- success probability/confidence `0.9`

Run separate grids for `--mode ours` and `--mode alpha` to save compute. Merge shards with `merge_utkface_bounded_vs_alpha_shards.py`, then analyze with `analyze_utkface_split_mode_results.py`.

## `(E,C)+M` Postprocessing

The `(E,C)+M` curve can be computed from saved `(E,C,G)+M` estimates without rerunning model inference:

```bash
python experiments/utkface_age/postprocess_utkface_ec_from_saved_estimates.py \
  --input_glob 'outputs/utkface_grid/ours/utkface_sigma*_alpha*_merged.json' \
  --output_dir outputs/utkface_grid/ours_with_ec \
  --ci-mode recompute_ec_2way \
  --ec-confidence 0.9 \
  --workers 8
```

## Convergence Plot

```bash
python experiments/utkface_age/utkface_single_point_convergence_analysis.py \
  --utk_dir data/UTKFace/UTKface_inthewild \
  --model_dir models/mivolo_v2_hf \
  --out_json outputs/utkface_single_point_convergence.json

python experiments/utkface_age/plot_utkface_appendix_convergence.py \
  --input outputs/utkface_single_point_convergence.json \
  --output outputs/utkface_appendix_convergence.pdf
```
