# Code Analysis: Paper 2278 — JiSi LLM Router

## Preparation Failure

### Root Cause
The embedding server (`gte-Qwen2-7B-instruct` on port 8000) was **not running** when the evaluation command executed. The `run_jisi.py` script calls `_generate_embeddings_concurrent()` which sends HTTP requests to `http://127.0.0.1:8000/v1/embeddings`. Without the server, all embedding calls fail with `Connection error`, eventually raising `RuntimeError: Received empty embedding from generator`.

### Repair Applied
Started the embedding server inside container `autosota_sota_paper_2278`:
```bash
cd /repo
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/models/models/iic--gte-Qwen2-7B-instruct/snapshots/master \
  nohup python3 embedding_server_local.py > /tmp/embedding_server.log 2>&1 &
```

### Verified Baseline
- Command: `python3 -m baselines.JiSi.run_jisi --config baselines/JiSi/config/jisi/main.router.json --output results/jisi/output.json`
- MMLU-Pro accuracy: **86.89%** (782/900) — matches manifest baseline exactly
- Overall accuracy: 67.65% (2236.5/3653 queries across 9 datasets)
- MMLU-Pro cost: $3.54
- Total cost: $17.42

## Container and Execution Environment

- Container: `autosota_sota_paper_2278` (image: `autosota/paper-2278:reproduced`)
- GPUs: 0,1 (both 80GB A100-class)
- Embedding server: GPU 0, port 8000, gte-Qwen2-7B-instruct
- Python: 3.10.13, PyTorch: 2.1.0, transformers: 4.44.2
- Repo: /repo, git at commit `0966497` tagged `_baseline`
- Data: /repo/data/jisi/seed42_split0.7/
- Models: /models/models/iic--gte-Qwen2-7B-instruct/snapshots/master/

## Key Code Architecture

### Routing Pipeline (run_jisi.py)
1. `build_embedding_bank()` (line ~417): Loads/builds embedding bank from training data
2. `route_queries_batch()` (line ~450): Primary routing — retrieves top-k similar training examples, computes per-model correctness scores, selects best model per query
3. `_generate_embeddings_concurrent()` (line ~383): Sends batched queries to embedding server
4. `evaluate_routing()` (line ~840): Orchestrates full evaluation loop
5. `run_router_mode()` (line ~920): Runs router-only evaluation

### Critical Hyperparameters (main.router.json)
- `rag_num`: 50 (support set size)
- `rag_thres`: 0.95 (similarity threshold)
- `weighted_score`: true
- `sample_n`: 3 (candidate models considered)
- Route scoring: weighted similarity × correctness on support examples

## Safe Optimization Targets

All ideas target internal routing logic or hyperparameters. No test data, labels, or evaluation protocol changes.

### Priority Order
1. **Idea 2 (P0)**: Discriminative Support Set Filtering — lowest risk, quickest implement
2. **Idea 1 (P0)**: Adaptive Support Retrieval via Local Embedding Density
3. **Idea 3 (P1)**: Per-Dataset Prior Bias for Router Scoring
4. **Idea 11 (P1)**: Coordinated rag_thres × rag_num Grid Sweep
5. **Idea 5 (P1)**: Query Difficulty-Adaptive Support Set Size
6. **Idea 8 (P0)**: Embedding Normalization Consistency Audit
7. **Idea 4 (P1)**: ISP-Based Tiebreaker
8. **Idea 6 (P2)**: Per-Model Temperature Scaling
9. **Idea 7 (P0)**: Response Length Score Normalization Fix
10. **Idea 9 (P1)**: Answer Extraction Audit

## Commit Strategy
- Each successful implementation committed with descriptive message
- `_best` tag updated only on Pareto-better results
- All scores recorded via `/tools/record_score.sh`
