# §4.5 — Scalability

Re-runs the §4.3 real-workloads pipeline ([`../real_workloads/`](../real_workloads/))
across additional GPU platforms and model architectures.

## §4.5.1 — Across GPU platforms (Table 6)

Run `real_workloads/run_grid_search.sh` on L40S and B300 (in addition to the
RTX PRO 6000 / H200 results already in Tables 2-3). Combine the resulting
`optimal_per_scheduler.json` files into Table 6.

```bash
cd ../real_workloads
MODEL=Qwen/Qwen3-8B GPUS=...  ./run_grid_search.sh    # on L40S box
MODEL=Qwen/Qwen3-8B GPUS=...  ./run_grid_search.sh    # on B300 box
```

Note: v0 results on B300 are unavailable because v0 does not support
PyTorch 2.9 / CUDA 13.0 (paper §4.5.1 footnote).

## §4.5.2 — Across model architectures (`scalmodel.pdf`)

Same pipeline, on RTX PRO 6000 with WildChat, for:
- Llama-3.1-8B-Instruct
- Mathstral-7B-v0.1
- Qwen2.5-Coder-7B
- DeepSeek-R1-Distill-Qwen-7B

```bash
cd ../real_workloads
for M in meta-llama/Llama-3.1-8B-Instruct mistralai/Mathstral-7B-v0.1 \
         Qwen/Qwen2.5-Coder-7B-Instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; do
    MODEL=$M ./run_grid_search.sh    # ~6 h each
    python analyze_grid_search.py outputs/RTXPRO6000_${M##*/}/
done

cd ../scalability
python plot_scalmodel.py \
    --inputs ../real_workloads/outputs/RTXPRO6000_*/optimal_per_scheduler.json \
    --output scalmodel.pdf
```

`plot_scalmodel.py --demo` renders an approximate version from the
illustrative numbers reported in the paper §4.5.2 table.

## §4.5.3 — Robustness / sensitivity

Computed from the same grid-search data; no separate runs needed.
The analysis is in
[`real_workloads/analyze_grid_search.py`](../real_workloads/analyze_grid_search.py)
(CV, range ratio, B-Sens, N-Sens metrics; see paper Appendix sensitivity).
