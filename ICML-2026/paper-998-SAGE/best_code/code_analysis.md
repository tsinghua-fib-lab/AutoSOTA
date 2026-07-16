# Code Analysis — Paper 998 (SAGE: Detecting Errors in AI-Generated Annotations)

## Evaluation Path
- Command: python3 tools/llm_analysis.py --dataset TruthfulQA --model qwen
- Input files in outputs/llm_generation/
- Output: stdout with AUROC values per method

## Pipeline Stages
1. Answer Generation: run_evaluation_llm_vllm.py with Qwen3-8B via vLLM
2. Judge Evaluation: judge_llm_pipeline.py with Qwen2.5-7B-Instruct on port 8999
3. KNN: knn_question_only.py with Qwen3-8B embeddings
4. Scoring: neighbor_based_llm_evaluator_vllm.py via vLLM
5. Analysis: llm_analysis.py computes AUROC per method

## Key Files for Optimization
| File | Role | Safe to modify? |
|------|------|-----------------|
| tools/llm_analysis.py | Metric computation, score aggregation | Yes |
| tools/neighbor_based_llm_evaluator_vllm.py | Scoring prompts, temperature | Yes |
| tools/knn_question_only.py | KNN parameters | Yes |
| tools/knn_qa.py | Q+A KNN alternative | Yes |

## Risky Files (DO NOT MODIFY)
- Judge files: ground truth labels
- run_evaluation_llm_vllm.py: answer generation
- judge_llm_pipeline.py: judge model
- AUROC computation in compute_auroc(): metric definition
- load_ground_truth(): ground truth loading

## Current Baseline
- SAGE_AUROC: 67.30% (paper: 72.61%)
- Direct_AUROC: 63.35% (paper: 65.34%)
- Random_AUROC: 68.43% (paper: 69.02%)

## Bottleneck Analysis
- Temperature=0 causes score collapse
- TRUTHFULQA_NEIGHBOR_PROMPT neutered
- Self-reference always included
