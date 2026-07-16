# Test-Time Model Querying - ResMAT1

This directory queries LLMs to generate data for **$P^{\text{testtime}}_1$** (irsl_testtime_resmat1).

## Query Configuration

- **Sampling**: 10,000 samples per question
- **Prompting**: Few-shot examples included
- **Temperature**: 1.0
- **Models**: 8 LLMs

## Scripts

- `monkey_query.py` - Main query script for LLM inference
- `monkey_query_pre.py` - Pre-processing for queries
- `monkey_query_utils.py` - Utility functions (prompt creation, answer matching)

## Usage

```bash
cd monkey/monkey_query
python monkey_query.py --model_nickname <model_name> --dataset <benchmark> --num_samples 10000
```

## Output

Query results are uploaded to `stair-lab/monkey_queries` on HuggingFace, which is then processed by `monkey_gather_data/` to create the final dataset.

## Related

- Data gathering: [`../monkey_gather_data/`](../monkey_gather_data/)
- Analysis: [`../monkey_analysis/`](../monkey_analysis/)
