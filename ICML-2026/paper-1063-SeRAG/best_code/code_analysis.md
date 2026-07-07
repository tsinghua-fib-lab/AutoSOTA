# SOTA Preparation Repair — Paper 1063 (SeRAG)

## Original Failure

The SOTA preparation failed because the container `autosota_sota_paper_1063` was started without `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables. The evaluation command references these as `${OPENAI_API_KEY}` and `${OPENAI_BASE_URL}`, but `docker exec` does not inherit host environment variables.

Error: `openai.OpenAIError: Missing credentials. Please pass an api_key or set the OPENAI_API_KEY environment variable.`

## Repair

The fixed evaluation command passes credentials inline in the `docker exec` invocation:

```
OPENAI_API_KEY="sk-..." OPENAI_BASE_URL="https://api.deepseek.com" \
  python3 run.py --spacy_model en_core_web_trf --embedding_model /models/all-MiniLM-L6-v2 \
  --dataset_name 2wikimultihop --llm_model deepseek-chat --max_workers 16
```

The DeepSeek API key is sourced from `config.yaml` (`deepseek_api_key` field). The `LLM_Model` class in `src/utils.py` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment, so any OpenAI-compatible provider works.

## Baseline Verification

| Metric | Reproduction Baseline | This Run (Iter 0) | Delta |
|--------|----------------------|-------------------|-------|
| Str-Acc (Contain Accuracy) | 78.6% | 78.7% | +0.1pp |
| LLM-Acc (LLM Accuracy) | 80.9% | 81.5% | +0.6pp |

Both accuracy metrics are within expected LLM non-determinism (DeepSeek-chat). The pipeline runs end-to-end: NER → graph construction → SE optimization → token-free consolidation → self-query → retrieval → QA → evaluation.

## Container State

- **Name**: `autosota_sota_paper_1063` (running, from `autosota/paper-1063:reproduced`)
- **GPU**: devices 2,3 (`CUDA_VISIBLE_DEVICES=0` inside container → first assigned GPU)
- **Git**: repo at `/repo`, tag `_baseline` at `b1e1293`, tag `_best` at `b69b073f`
- **Tools**: `/tools/record_score.sh` present and executable
- **Artifacts**: `/autosota_artifacts/paper-1063/sota/` writable

## Key Source Files

| File | Role |
|------|------|
| `run.py` | Entry point: loads data, creates SeRAG, runs index→qa→evaluate |
| `src/config.py` | `SeRAGConfig` dataclass with all hyperparameters |
| `src/SeRAG.py` | Core RAG class: `index()`, `qa()`, `retrieve()`, graph construction, matching |
| `src/struct_entropy.py` | Structural entropy computation for encoding tree |
| `src/embedding_store.py` | Embedding storage/retrieval with parquet backend |
| `src/evaluate.py` | Evaluator class computing LLM-Acc and Str-Acc |
| `src/utils.py` | `LLM_Model` class (OpenAI client wrapper), normalization |

## Key Hyperparameters (SeRAGConfig defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `retrieval_top_k` | 3 | Final retrieved chunks (paper optimal: 7) |
| `k_dim` | 2 | Entropy dimension for tree |
| `gamma_coarse` | 0.4 | Coarse-grained matching weight |
| `gamma_fine` | 0.6 | Fine-grained matching weight |
| `retrieval_k_coarse` | 10 | Coarse-grained communities to consider |
| `alpha_semantic` | 0.45 | Semantic edge weight |
| `beta_logical` | 0.45 | Logical edge weight |
| `gamma_distance` | 0.10 | Distance edge weight |
| `semantic_k` | 20 | KNN for semantic edges |

## Safe Optimization Targets

1. **Config-only**: `retrieval_top_k`, `gamma_coarse/fine`, `k_dim` — no code changes
2. **Retrieval post-processing**: filter/dedup chunks after `retrieve()` in `qa()`
3. **Self-query prompt**: prompt engineering in `_batch_llm_self_query()`
4. **Coarse-grained matching threshold**: in `_coarse_grained_matching()`
5. **Entity matching heuristics**: in `get_seed_entities()`
