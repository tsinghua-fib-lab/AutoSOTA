# Code Analysis for Paper 3454 - CORRECT SOTA Optimization

## Evaluation Path
- **Inference**: `src/inference_whoandwhen.py` → `modify_analyze_all_at_once_local()` → `_run_local_generation()` in `src/Lib/local_model.py`
- **Evaluation**: `src/evaluate.py` → `read_predictions()` parses output → `evaluate_accuracy()` computes step accuracy with tolerance
- **Metric Parsing**: Regex `r"Agent Name:\s*([\w_]+)"` and `r"Step Number:\s*(\d+)"` in `evaluate.py:31-32`
- **Output**: `outputs_whoandwhen/correct_hc_k10.txt` (from eval command)

## Inference Flow
1. `inference_whoandwhen.py` loads schemata and similarities
2. Creates `SimilarityBasedSchemaAnalyzer` with schemata + similarities
3. `modify_analyze_all_at_once_local()` wraps `analyze_all_at_once_local()`
4. For each JSON in data directory:
   a. Builds original prompt with problem, answer, chat history
   b. `modify_prompt()` injects k=10 most-similar schemata + instructions
   c. Calls `_run_local_generation()` with temperature=0.0, do_sample=False
   d. Gets response, writes to output file

## Key Files
- `src/inference_whoandwhen.py` - Main entry point, schema analyzer, prompt modification
- `src/Lib/local_model.py` - `_run_local_generation()`, `analyze_all_at_once_local()`
- `src/evaluate.py` - Prediction parsing and accuracy computation
- `src/generate_trajectory_similarities.py` - Similarity precomputation (BAAI/bge-m3)
- `data/schemata_whoandwhen/Hand-Crafted/error_schemata.txt` - 10 error schemata
- `data/similarities_whoandwhen/Hand-Crafted_trajectory_similarities.json` - Precomputed similarities

## Safe Modification Targets
1. **Prompt template** in `SimilarityBasedSchemaAnalyzer.modify_prompt()` (line 285-393) - safe, only changes LLM instructions
2. **Schema selection** in `get_similarity_based_schema()` (line 222-283) - safe, only changes which schemata are retrieved
3. **Output parsing** in `src/evaluate.py` (line 31-32) - safe, only adds fallback patterns
4. **Similarity precomputation** in `src/generate_trajectory_similarities.py` - safe offline change
5. **Schema file** `data/schemata_whoandwhen/Hand-Crafted/error_schemata.txt` - safe, appending new schemata
6. **Temperature and num_schemata parameters** - safe parameter changes

## Risky Files (DO NOT MODIFY)
- `src/evaluate.py` metric computation logic (line 76-127) - evaluation protocol
- Data files in `data/whoandwhen/Hand-Crafted/` - test data/labels
- `/tools/record_score.sh` - scoring infrastructure
- `_baseline` and `_best` tags - git state

## Prompt Template Locations
- Original prompt (baseline): `analyze_all_at_once_local()` line 635-648 in `Lib/local_model.py`
- Schema-enhanced prompt: `modify_analyze_all_at_once_local()` line 463-476 in `inference_whoandwhen.py`
- Schema injection + instructions: `modify_prompt()` line 285-393 in `inference_whoandwhen.py`

## Key Parameters
- `temperature=0.0` - deterministic generation in `_run_local_generation()` line 103
- `num_schemata=10` - eval command uses `--num_schemata 10`
- `max_tokens=1024` - max generation tokens
- BAAI/bge-m3 for embeddings (in `generate_trajectory_similarities.py`)
