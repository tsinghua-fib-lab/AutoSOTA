# §4.4 + Appendix — PD disaggregation comparison

Compares EB⁺ to vLLM's built-in disaggregation scheduler (DistServe /
Splitwise) on 2- and 4-GPU setups.

## Files

| File | Purpose |
|---|---|
| `disagg_multi_proxy.py` | Multi-backend disaggregation proxy (routes prefill / decode to separate vLLM instances) |
| `run_disagg_baseline.sh` | Stand up the disagg baseline with a fixed P:D ratio |
| `run_2gpu_comparison.sh`, `run_4gpu_comparison.sh` | EB⁺ vs disagg across P:D ratios |

## Run

```bash
./run_2gpu_comparison.sh         # ~2-3 h
./run_4gpu_comparison.sh         # ~3-4 h
./run_disagg_baseline.sh         # ~1 h
```

EB⁺ runs on a single vLLM instance; the disagg baselines run two coordinated
instances behind `disagg_multi_proxy.py`. Output JSONs are compared
post-hoc — there is no aggregated plot script in the paper for this section
(numbers appear inline in §4.4 and the appendix table).
