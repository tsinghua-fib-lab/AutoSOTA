# Multi-turn dialogue (WildChat) preprocessing

WildChat is the multi-turn dialogue dataset used in §4.3.2. Each user
session is a sequence of turns; the harness replays turns at controlled
concurrency.

## Files

| File | Purpose |
|---|---|
| `export_dataset.py` | Pull WildChat from HuggingFace, filter to multi-turn |
| `run_benchmark.sh`, `run_concurrency_sweep.sh` | Replay sessions and benchmark |
| `analyze_results.py` | Aggregate per-session statistics |

## Run

```bash
python export_dataset.py --output wildchat_multiturn.jsonl
./run_benchmark.sh
python analyze_results.py outputs/
```
