# §3 — Kernel breakdown

Measures attention kernel time as a function of token count across three
configurations: `pure_prefill`, `pure_decode`, and `mixed` (parameterised
by decode percentage). Backs `kernel_breakdown*.pdf` in §3 of the paper.

## Files

| File | Purpose |
|---|---|
| `benchmark_flash_attn_sweep.py` | FlashAttention timing benchmark (canonical sweep used for the paper) |
| `results_h200.json`, `results_a6000.json` | Committed results from the paper |
| `plot_kernel_breakdown.py` | Render the figure(s) |

## Run

```bash
# To regenerate on your hardware (canonical sweep — output schema matches
# what plot_kernel_breakdown.py expects):
python benchmark_flash_attn_sweep.py --output results_$(hostname).json

# To render from existing JSON(s):
python plot_kernel_breakdown.py \
    --inputs results_h200.json results_a6000.json \
    --output kernel_breakdown.pdf
```

## Expected output

Two-panel log-log plot (one panel per GPU), each showing `pure_prefill`,
`pure_decode`, and `mixed` curves at 10/20/40/80% decode token share.
