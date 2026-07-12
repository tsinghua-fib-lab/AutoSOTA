# §4.4 — EB⁺ non-stationary workloads (Table 5)

Two scenarios:
- **Distribution shift**: `(μ_L, μ_O) = (1024, 128) → (512, 512) → (128, 1024)`,
  2k requests per phase, `c = 2048`.
- **Concurrency shift**: `c = 32 → 512 → 1024 → 256 → 2048`.

## Files

| File | Purpose |
|---|---|
| `generate_distribution_shift_dataset.py` | Synthesize the phased dataset |
| `run_distribution_shift.sh`, `run_concurrency_shift.sh` | Run v1 / EB / EB⁺ on each phase |
| `plot_distribution_shift.py` | Per-phase throughput / TTFT / TPOT timeline |

## Run

```bash
python generate_distribution_shift_dataset.py   # one-time, ~5 min
./run_distribution_shift.sh                     # ~1 h
./run_concurrency_shift.sh                      # ~1 h
python plot_distribution_shift.py outputs/
```
