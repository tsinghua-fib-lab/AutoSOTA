# raw_data

This directory ships three zip archives — one per model — that together contain
the 30 watermarked-pivotal pickles consumed by the analysis scripts:

```
opt1.3b_data.zip                10 pickles (Gumbel × 5 temps + Inverse × 5 temps)
qwen2p5_3b_data.zip             10 pickles
sheared_llama_2p7b_data.zip     10 pickles
```

Unpack them in place before running anything:

```bash
cd raw_data && for z in *_data.zip; do unzip -o "$z"; done && cd ..
```

This drops 30 `.pkl` files into `raw_data/`. Filenames follow the pattern

```
{model}{_inv?}_temp_{T}_len_{m}_cnt_{N}.pkl
```

with `model ∈ {opt1.3b, qwen2p5_3b, sheared_llama_2p7b}`, `_inv` present for
the inverse-transform watermark and absent for Gumbel-max, `T ∈ {0.1, 0.3,
0.5, 0.7, 0.9}` the decoding temperature, `m = 400` tokens per document, and
`N = 1000` documents per file for Gumbel (`N = 500` for Inverse).

Each pickle stores a dict; the only field the analysis scripts read is

```python
data["watermark"]["Ys"]   # (N, m) token-level pivotal statistics Y
```

For the Gumbel-max watermark, `Y` lies in `[0, 1]` and is Uniform(0, 1) under
the null, so the goodness-of-fit tests are applied directly. For the
inverse-transform watermark, `Y = -|U - η|` lies in `[-1, 0]`; the analysis
scripts pass `|Y|` through its null CDF `F(r) = 1 - (1 - r)²` to obtain a
Uniform(0, 1) sample, then apply the same tests.
