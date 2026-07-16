# Test-Time Model Querying - ResMAT2

This directory queries LLMs to generate data for **$P^{\text{testtime}}_2$** (irsl_testtime_resmat2).

## Query Configuration

- **Sampling**: 2,560 samples per question
- **Prompting**: Zero-shot with chain-of-thought
- **Temperature**: 0.6
- **Models**: 12 latest LLMs (DeepSeek-R1, Qwen3, etc.)

## Scripts

- `query.py` - Main query script for LLM inference
- `query_utils.py` - Utility functions (zero-shot prompts, answer matching)
- `query_llama.sh` - Shell script for Llama model queries
- `query_small.sh` - Shell script for smaller models
- `query_smoke_test.py` - Quick validation test

## Usage

```bash
cd monkey/monkey_query_2

# Using Python script
python query.py --model_nickname <model_name> --dataset <benchmark> --num_samples 2560

# Using shell scripts
bash query_llama.sh
bash query_small.sh
```

## Output

Query results are uploaded to `stair-lab/denoise_eval_query` on HuggingFace, which is then processed by `monkey_gather_data_2/` to create the final dataset.

## Differences from ResMAT1

- Zero-shot prompting (no few-shot examples)
- Chain-of-thought reasoning enabled
- Lower temperature (0.6 vs 1.0) for more focused responses
- Includes shell script wrappers for batch execution
- Includes smoke test for validation

## Related

- Data gathering: [`../monkey_gather_data_2/`](../monkey_gather_data_2/)
- Analysis: [`../monkey_analysis/`](../monkey_analysis/)
