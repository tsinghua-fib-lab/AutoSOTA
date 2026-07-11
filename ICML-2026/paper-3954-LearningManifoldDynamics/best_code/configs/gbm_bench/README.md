Synthetic GBM end-to-end benchmark configs for BNRDE (`bnrde`).

Run the full sweep with:

## Shuffle
```bash
for cfg in configs/gbm_bench/shuffle/*.toml; do
  uv run train --config "$cfg"
done
```


## GL
```bash
for cfg in configs/gbm_bench/gl/*.toml; do
  uv run train --config "$cfg"
done
```

## MKW
```bash
for cfg in configs/gbm_bench/mkw/*.toml; do
  uv run train --config "$cfg"
done
```