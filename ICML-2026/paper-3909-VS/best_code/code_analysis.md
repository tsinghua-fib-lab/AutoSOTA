# Code Analysis for Paper 3909: Verbalized Sampling

## Evaluation Path
- Entry: `eval_reproduction.py` → `Pipeline.run_complete_pipeline()`
- Generation: `pipeline.py` → uses `PromptFactory.get_prompt()` + LLM backend
- Selection: `selection.py` → `postprocess_responses()` with tau filtering
- Evaluation: `analysis/evals/diversity.py`, `ngram.py`, `length.py`

## Key Files

### Generation
- `verbalized_sampling/pipeline.py` — Main pipeline orchestrator
- `verbalized_sampling/methods/factory.py` — Method definitions (Method enum), PromptFactory
- `verbalized_sampling/methods/prompt.py` — Prompt templates per task type (BasePromptTemplate, CreativityPromptTemplate, etc.)
- `verbalized_sampling/methods/parser.py` — ResponseParser for parsing LLM outputs
- `verbalized_sampling/selection.py` — DiscreteDist, postprocess_responses() for candidate selection
- `verbalized_sampling/llms/openai.py` — OpenAI-compatible LLM backend (used for DeepSeek)
- `verbalized_sampling/llms/__init__.py` — LLM registry and routing

### Evaluation
- `verbalized_sampling/analysis/evals/diversity.py` — DiversityEvaluator: embeddings + cosine similarity
- `verbalized_sampling/analysis/evals/ngram.py` — NgramEvaluator: Rouge-L, Distinct-N
- `verbalized_sampling/analysis/evals/length.py` — LengthEvaluator: token/word counts
- `verbalized_sampling/llms/embed.py` — Embedding models: OpenAIEmbeddingModel, LocalEmbeddingModel (sentence-transformers), TfidfEmbeddingModel (HashingVectorizer)

### Config
- `eval_reproduction.py` — CLI entry point with all tunable parameters

## Baseline Metrics (iteration 0)
- Diversity_VS_Standard: 27.12%
- Diversity_Direct: 26.43%
- RougeL_VS_Standard: 16.67%
- RougeL_Direct: 17.35%

## Metric Parsers
- Diversity: `diversity_results.json` → `overall_metrics.avg_diversity * 100`
- RougeL: `ngram_results.json` → `overall_metrics.avg_rouge_l * 100`
- Length: `length_results.json` → `overall_metrics.avg_token_length`

## Known Levers (from manifest)
1. `--num-prompts`: 5 (paper: 100)
2. `--model`: DeepSeek v4-flash (paper: GPT-4.1-Mini)
3. `temperature`/`top_p`: 0.7/1.0 in eval_reproduction.py (paper: 0.7/1.0)
4. `k/N`: k=5 candidates, N=30 responses
5. Probability tuning: lower threshold → higher diversity (supported in code but not exposed)
6. VS variants: VS-CoT, VS-Multi
7. Embedding model: hash-based (can switch to sentence-transformers)

## Safe Modification Targets
- `eval_reproduction.py`: Add CLI args for probability_tuning, temperature, top_p, embedding model
- `verbalized_sampling/methods/prompt.py`: Improve creativity prompt templates for diversity
- `verbalized_sampling/methods/factory.py`: Expose probability_tuning parameter
- `verbalized_sampling/pipeline.py`: Pass new parameters through ExperimentConfig
- Environment variables: `LOCAL_EMBED_MODEL` for embedding model selection

## Risky Files (DO NOT MODIFY)
- `verbalized_sampling/analysis/evals/diversity.py` — Metric formula
- `verbalized_sampling/analysis/evals/ngram.py` — RougeL computation
- `verbalized_sampling/analysis/evals/length.py` — Length computation
- `verbalized_sampling/methods/parser.py` — Response parsing logic
- `verbalized_sampling/selection.py` — Selection algorithm

## Data Files
- `data/poem_titles.txt` — 246 poem prompts
- `/datasets/`, `/models/` — Cache mounts (no paper-specific data)

## Current State
- Container: autosota_repro_paper_3909
- Repo commit: 5418a1b (iter-0 baseline)
- Model: openai/deepseek-v4-flash (DeepSeek API via OpenAI-compatible endpoint)
- Embeddings: HashingVectorizer (384-dim hash-based)
- Working directory: /repo
