# LORD-GoF

Code and data for the ICML submission *LORD-GoF: A Robust Online Detection
Approach for LLM Watermarks in Sparse and Mixed Streams*.

This repository contains the analysis code and the watermarked-pivotal data
that the paper's figures and tables are built from. Generation is not included
here — the watermark generation pipeline follows
[Li et al. (2025)](https://arxiv.org/abs/2411.13868), and the pivotal
statistics produced by that pipeline are checkpointed under `raw_data/` so the
experiments can be re-run directly. If you'd like to regenerate the data from
scratch, see
[lx10077/TrGoF](https://github.com/lx10077/TrGoF) and
[hwq0726/GoF-for-Watermark-Detection](https://github.com/hwq0726/GoF-for-Watermark-Detection),
which provide the generation code on which our data is based.

## What's here

```
LORD-GoF/
├── StreamAnalysis.py          Figure 2 — cumulative FDR and Power over time
├── SparsityAnalysis.py        Figure 3 — FDR and Power as a function of global sparsity π
├── AttackAnalysis.py          Robustness tables — substitution / deletion / insertion edits
├── HyperparameterAnalysis.py  Sensitivity to LORD's (w₀, γ_exp)
├── raw_data/                  3 zip archives — 30 .pkl files in total (unzip before running)
│   ├── opt1.3b_data.zip
│   ├── qwen2p5_3b_data.zip
│   └── sheared_llama_2p7b_data.zip
├── requirements.txt
└── LICENSE
```

The four analysis scripts evaluate eight goodness-of-fit statistics
(Kolmogorov, Kuiper, Anderson–Darling, Cramér–von Mises, Watson, Chi-squared,
Rao, Greenwood; reported in the paper as Ney and Phi, respectively) under the online LORD procedure and a naive fixed-threshold
baseline, on each (model, watermark) configuration used in the paper.

## Installation

```bash
git clone <this-repo>.git LORD-GoF
cd LORD-GoF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The watermarked-pivotal data lives under `raw_data/` as three model-specific
zip archives (so each individual file stays well below GitHub's size limits).
Unpack them in place before running the analysis scripts:

```bash
cd raw_data && for z in *_data.zip; do unzip -o "$z"; done && cd ..
# or, if you have unzip-everything-at-once aliases:
# unzip 'raw_data/*_data.zip' -d raw_data/
```

This produces the 30 `.pkl` files (3 models × 2 watermarks × 5 temperatures)
that the analysis scripts read. The zips can be deleted after unpacking, or
kept around — they're listed in `.gitignore`'s sibling `.pkl` pattern so the
unpacked pickles won't be committed back.

Python 3.10 is what we tested with; 3.9–3.12 should also work.

## Reproducing the experiments

The five (model, watermark) configurations are defined inside each script and
match those in the paper, so the default commands just run them top to bottom:

```bash
# Figure 2 — stream dynamics
python StreamAnalysis.py
# writes my_plot/{model}_{gum,inv}_dyn_{fdr,pow}.pdf

# Figure 3 — impact of global sparsity
python SparsityAnalysis.py
# writes my_plot/{model}_{gum,inv}_pi_{fdr,pow}.pdf

# Robustness tables — human-edit attacks
python AttackAnalysis.py --model qwen --wm_type gum --temp 0.5
# loop over the five paper configurations:
for cfg in "opt inv 0.7" "qwen gum 0.5" "qwen inv 0.7" "llama gum 0.5" "llama inv 0.7"; do
    read m w t <<<"$cfg"
    python AttackAnalysis.py --model "$m" --wm_type "$w" --temp "$t"
done

# Hyperparameter ablation
python HyperparameterAnalysis.py
```

`AttackAnalysis.py` exposes the most useful flags: `--model {opt|qwen|llama}`,
`--wm_type {gum|inv}`, `--temp 0.1..0.9`, `--pi`, `--m`, `--alpha`.

## Data layout

Filenames in `raw_data/` follow a fixed pattern:

```
{model}{_inv?}_temp_{T}_len_{m}_cnt_{N}.pkl
```

with `model ∈ {opt1.3b, qwen2p5_3b, sheared_llama_2p7b}`, `_inv` present for
the inverse-transform watermark and absent for Gumbel-max, `T` the decoding
temperature, `m = 400` tokens per document, and `N` documents per file (1000
for Gumbel, 500 for Inverse).

Each pickle is a dict whose only field that the analysis scripts read is

```python
data["watermark"]["Ys"]   # shape (N, m), token-level pivotal statistics
```

For the Gumbel-max watermark, `Y ∈ [0, 1]` and is used directly. For the
inverse-transform watermark, `Y ∈ [-1, 0]` and is mapped to Uniform(0, 1) under
the null via `F(r) = 1 - (1 - r)²` inside each script.

## Method, briefly

For each document in the stream we read the token-level pivotal vector `Y`,
optionally mix in null tokens at local density `ρ` (per-document watermark
fraction), map `Y` to a uniform sample under H₀ by the probability integral
transform, and compute a document-level p-value for each goodness-of-fit
statistic via a semiparametric Monte-Carlo calibration. The p-value is then
fed to either the online LORD controller or a naive fixed-threshold baseline,
and we accumulate true and false discoveries over the stream to report FDR
and Power. The qualitative observation behind the figures is that the naive
baseline's FDR climbs toward one as the watermark becomes sparse, while LORD
keeps FDR at or below the target `α = 0.05` across the sparsity range without
giving up much power.

## Citation

```bibtex
@inproceedings{xu2026lordgof,
  title     = {LORD-GoF: A Robust Online Detection Approach for LLM Watermarks
               in Sparse and Mixed Streams},
  author    = {Xu, Jiade and Li, Zhouping},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}

@article{li2025robust,
  title   = {Robust detection of watermarks for large language models under
             human edits},
  author  = {Li, Xiang and Ruan, Feng and Wang, Huiyuan and Long, Qi and
             Su, Weijie J.},
  journal = {Journal of the Royal Statistical Society Series B:
             Statistical Methodology},
  year    = {2025}
}
```

## License

MIT (see `LICENSE`). The `.pkl` files in `raw_data/` are released for
academic reproduction of the experiments reported in the paper.
