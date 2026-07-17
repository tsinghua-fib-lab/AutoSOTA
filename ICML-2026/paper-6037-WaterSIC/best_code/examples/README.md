# Examples

Auxiliary scripts that aren't part of the documented quantize / eval / finetune workflow.

## `paper/`

Diagnostic plots, ablations, and one-off analyses used while developing the paper. None of these are called from the main pipeline; they consume already-produced run directories or pre-cached metrics. Kept here for reproducibility of paper figures.

| Script | Purpose |
|---|---|
| `analyze_activation_outliers.py` | Find input features with anomalously large norms (motivated `--zero_out_rows`). |
| `compute_activation_mse.py` | Cache per-layer activation MSE / cosine-distance between BF16 and quantized models. Output `.pt` files are consumed by `scripts/plot_activation_mse.py`. |
| `plot_rate_distortion_1B.py` | Generate the Llama-3.2-1B rate-distortion curve figure. |
| `plot_zsic_column_entropies.py` | Per-in-channel entropy histogram (early exploratory version). |
| `plot_zsic_column_entropies_paper.py` | Camera-ready version of the per-in-channel entropy histogram (Appendix figure). |
| `scan_w2_input_norms.py` | Scan `w2`-input column norms to identify outlier dimensions. |
| `test_zero_multi_layers_ppl.py` | Sanity check: zeroing specific rows in the unquantized model vs PPL impact. |
| `submit_pipeline_jobs.py` | Personal batch launcher (uses `scheduler.py`). |
| `sweep_finetune_zsic.py` | Multi-GPU finetune sweep wrapper around `scripts/finetune_zsic.py`. |

Run from the repo root, e.g.:

```bash
python -m examples.paper.plot_rate_distortion_1B --run_root $QUANT_BUCKET/quant_runs/3.2-1B
```
