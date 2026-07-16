# Test-Time Data Collection - ResMAT1

This directory generates **$P^{\text{testtime}}_1$** (irsl_testtime_resmat1) used in the ICML paper.

## Dataset Specifications

- **Models**: 8 LLMs
- **Questions**: 552 questions from 6 benchmarks
- **Max Samples**: 10,000 per question
- **Prompting**: Few-shot, temperature 1.0
- **Benchmarks**: commonsense, legalbench, med_qa, bbq, lsat_qa, legal_support

## Data Source

- Downloads from: `stair-lab/monkey_queries` (HuggingFace)
- Uploads to: `stair-lab/irsl_testtime_resmat1` (HuggingFace)

## Usage in Paper

- **Section**: 4.2 (Test-time Item Response Scaling Law)
- **Figures**: Appendix validation figures (z-score correlation with HELM)
- **Purpose**: Validation against HELM benchmarks, demonstrates generalization

## Models Included

- Meta-Llama-3-8B-Instruct
- Meta-Llama-3-70B-Instruct
- Qwen3-8B, Qwen3-14B, Qwen3-32B
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-V2-Lite-Chat
- gemma-3-27b-it

## Scripts

- `testtime_gather_3d_resmat.py` - Main data gathering script
- `monkey_gather_we_query.py` - Process WE query results
- `monkey_gather_we_query_stats.py` - Statistics computation
- `monkey_query_fix_we_query.py` - Fix/patch query results
- `monkey_gather_harmbench.py` - HarmBench data gathering

## Generated Dataset Structure

```
Shape: (8 models, 552 questions, 10000 samples)
All entries have 10,000 samples (fully dense)
```
