# Paper 12 — EfficientQAT

**Full title:** *EfficientQAT: Efficient Quantization-Aware Training for Large Language Models*

**Registered metric movement (internal ledger / `results.md`, ASCII):** WikiText-2 PPL **7.1654 → 6.73** (**−6.08%**, lower is better). C4 PPL **8.9043 → 8.95** (slightly worse; headline metric is WikiText-2).

**Target:** WikiText-2 PPL ≤ **7.0217** (≈2% vs baseline **7.1654**) — **met**.

**Latest pipeline run:** `run_20260323_212440` (see `auto-pipeline/optimizer/papers/paper-152/runs/`).

---

## What changed (high level)

The int2 EfficientQAT checkpoint keeps **integer `qweight` / `qzeros` fixed**; gains came from **post-hoc calibration** of **per-layer quantization `scales`** and **RMSNorm `weight`** by minimizing cross-entropy on a **small slice of WikiText-2 train** (not test), with Adam.

- **Iter 1:** FP32 dequantization path — no gain (bottleneck is int2 weights).
- **Iter 2:** Light scale calibration (**3 steps**, LR **1e-5**, **16** samples) — small PPL drop (**7.1654 → 7.1485**).
- **Iter 3 (winning):** Stronger calibration — **10 steps**, LR **5e-5** (scales) / **2e-5** (norms), **32** samples × 2048 tokens, cosine LR, grad clip — **WikiText-2 PPL 6.73**.

Supporting inference tweaks (from the same run): **bfloat16** for stability, **SDPA** attention, and **auto-load** of sidecar tensors `{model}-scales.pt` and `{model}-norms.pt` after `QuantLinear` init (`quantize/int_linear_real.py`). Calibration driver scripts live at repo root as `calibrate_scales.py` / `calibrate_scales_v2.py`.

**Red lines honored:** eval protocol unchanged (`ppl_seqlen=2048`, full WikiText-2 test, C4 windows/seed); only scales/norms trained; integer buffers untouched; calibration data = **train** split only.

---

## Files to read first

| Area | Path |
|------|------|
| Quant linear + scale load | `quantize/int_linear_real.py` |
| Block-AP entry / PPL args | `main_block_ap.py` |
| Calibration | `calibrate_scales.py`, `calibrate_scales_v2.py` |

---

## Reproduce / sanity-check

Use the project’s documented eval commands for **Llama-2-7B w2g128** EfficientQAT; ensure calibrated **`scales` / `norms`** files sit beside the model path expected by `int_linear_real.py`. WikiText-2 PPL should align with **~6.73** under the frozen rubric settings.
