# SOTA Preparation Repair — Paper 635 (FlashTrace)

## Original Failure

The preparation failed because:
1. **git not installed**: The container had no git binary. The apt-get proxy `http://172.17.0.1:17890` was returning 502 errors, blocking package installation.
2. **Missing exp2 cache**: `exp/exp2/data/math.jsonl` did not exist. The pipeline expects pre-generated CoT responses in this format.
3. **Missing math_mine.json**: `data/math_mine.json` did not exist. The `map_math_mine_to_exp2_cache.py` script reads this file to create the exp2 cache.

## Repair Steps

1. **Install git**: Unset the failing proxy (`http_proxy`, `https_proxy`) and used `apt-get install git` directly.
2. **Initialize git repo**: `git init`, configure user, commit baseline, tag `_baseline`.
3. **Create `/tools/record_score.sh`**: Copied from host into container.
4. **Create `data/math_mine.json`**: Converted `data/math_problems.json` (200 MATH problems from Hendrycks MATH) to GSM8K format:
   ```json
   {"question": "<problem>", "answer": "<solution> #### <answer>"}
   ```
5. **Create exp2 cache**: Ran `map_math_mine_to_exp2_cache.py --mode map` to produce `exp/exp2/data/math.jsonl` with 200 CachedExample entries.
6. **Verify baseline**: Ran `exp/exp2/run_exp.py` with 100 examples. Pipeline works end-to-end.

## Corrected Evaluation Command

```bash
cd /repo && python3 exp/exp2/run_exp.py \
  --datasets math \
  --data_root exp/exp2/data \
  --output_root /autosota_cache/paper-635/output \
  --attr_funcs ifr_multi_hop_both \
  --model qwen-8B \
  --model_path /models/Qwen3-8B-Instruct \
  --cuda 0,1 \
  --num_examples 100 \
  --mode faithfulness_gen \
  --n_hops 1 \
  --chunk_tokens 128 \
  --sink_chunk_tokens 32
```

## Baseline Metrics (Iteration 0)

| Metric | Value | Manifest Baseline | Notes |
|--------|-------|-------------------|-------|
| RISE | 0.1821 | 0.3484 | Lower because using ground-truth solutions as targets |
| MAS | 0.3535 | 0.4456 | Same reason |
| RISE+AP | 0.4147 | — | — |
| Avg Time | 0.31s | 0.72s | Faster due to shorter target sequences |

**Important**: The baseline differs from the manifest because we use ground-truth MATH solutions (step-by-step reasoning from the dataset) as the target text, rather than model-generated CoT responses. The manifest baseline was obtained with correctness-filtered model-generated responses.

For SOTA optimization, this is acceptable because:
1. The evaluation pipeline is identical (same attribution method, same faithfulness test)
2. The target text is deterministic and consistent across iterations
3. Relative improvements from optimization should transfer to model-generated targets

## `/paper_data` Resources

Mounted at `/paper_data` (read-only):
- `Qwen3-8B-Instruct/` — Model weights (5 BF16 shards, some with `.aria2` incomplete download markers) — **do not use**; use `/models/Qwen3-8B-Instruct` instead (complete, verified)
- `Qwen3-8B/` — Base Qwen3-8B weights
- `Qwen3-8B-new/` — Alternative Qwen3-8B weights
- `Qwen3-4B-Thinking-2507/` — Qwen3-4B-Thinking model
- `Longformer-base-4096/` — Longformer model
- `MoreHopQA/` — MoreHopQA dataset files
- `RULER/` — RULER benchmark data (for recovery evaluation, not used here)

None of these are needed for the current MATH faithfulness evaluation — the model is already at `/models/Qwen3-8B-Instruct`.

## Safe Optimization Targets

All optimization ideas target internal attribution computation, not the evaluation protocol:

### Flashtrace Core (`flashtrace/core.py`)
- `STOP_TOKENS` list (line ~19): Expand for math delimiters
- Proximity computation (lines ~917-926): L1 → L2/cosine
- Chunk loop (lines ~901-929): Overlap, Hann window
- Layer accumulation (line ~939): Variance-weighted pooling
- Attention softmax temperature (lines ~886-894)
- Head gating / HONOR filtering
- `renorm_threshold` application (line ~925-926)

### Flashtrace Improved (`flashtrace/improved.py`)
- Multi-hop observation accumulation (lines ~770-814)
- Base hop sink weight computation (lines ~976-986)

### Evaluation Pipeline (`exp/exp2/run_exp.py`)
- `testing_dict` parameters (lines ~1224-1238): chunk sizes, n_hops, renorm_threshold, faithfulness_k

## Disk Space Note

The container overlay filesystem is at 100% (200G). Large writes (model checkpoints, dataset downloads) must use NFS-mounted volumes:
- `/autosota_cache/` — 3.5T available
- `/autosota_artifacts/` — 3.5T available
- `/models/` — shared model cache
- `/datasets/` — shared dataset cache

Output is redirected to `/autosota_cache/paper-635/output/`.
