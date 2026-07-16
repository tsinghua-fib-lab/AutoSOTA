# Test-Time Data Collection - ResMAT2

This directory generates **$P^{\text{testtime}}_2$** (irsl_testtime_resmat2) used in the ICML paper.

## Dataset Specifications

- **Models**: 12 latest LLMs
- **Questions**: 120 questions from 4 latest benchmarks
- **Max Samples**: 2,560 per question (paper reports ~2,500)
- **Prompting**: Chain-of-thought, temperature 0.6, zero-shot
- **Benchmarks**: AIME 2024, AIME 2025, MMLU Pro, Global MMLU Lite

## Data Source

- Downloads from: `stair-lab/denoise_eval_query` (HuggingFace)
- Uploads to: `stair-lab/irsl_testtime_resmat2` (HuggingFace)

## Usage in Paper

- **Section**: 4.2 (Test-time Item Response Scaling Law)
- **Figures**: Main paper figures with recent models (DeepSeek-R1, AIME experiments)
- **Purpose**: Evaluate latest models on challenging benchmarks

## Models Included

- DeepSeek-R1-Distill-Llama-70B
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Qwen-7B/14B/32B
- QwQ-32B
- Qwen3-4B/8B/14B/30B-A3B/32B
- gemma-3-27b-it

## Excluded Models (Data Quality)

12 models excluded due to incomplete data:
- Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct
- Mistral-7B-Instruct-v0.3, Mistral-Small-3.2-24B-Instruct-2506
- Phi-4-mini-instruct, Phi-4-mini-reasoning, Phi-4-reasoning-plus
- gemma-3-12b-it, gemma-3-4b-it
- SmolLM3-3B
- OLMo-2-1124-7B/13B-Instruct, OLMo-2-0325-32B-Instruct

## Excluded Questions

19 MMLU Pro prompts excluded (incomplete across models):
- prompt=106, 711, 1519, 1584, 1674, 2547, 2615, 3527, 4333, 4552, 4557, 5514, 5574, 5635, 5881, 6224, 6924, 9654, 9891, 11438

## Scripts

- `gather.py` - **Main unified gathering script**
- `show_progress.py` - Progress visualization and monitoring
- `gen_exclude_qpairs.py` - Utility to identify problematic question pairs

## Generated Dataset Structure

```
Shape: (12 models, 120 questions, 2560 samples)
Actual sample counts:
  - 1,423 entries with 2,560 samples (98.8%)
  - 17 entries with 2,048 samples (1.2%)
Mean: 2,554 samples per entry
```

## Note on Sample Count

⚠️ **Minor discrepancy**: The paper reports ~2,500 samples, but the code has `max_samples = 2560` and most entries contain exactly 2,560 samples. This is a rounding difference and does not affect the scientific conclusions.
