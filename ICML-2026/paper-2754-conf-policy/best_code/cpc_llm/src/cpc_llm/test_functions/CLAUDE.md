# test_functions/

Utilities for the Ehrlich protein motif discovery test function used as the primary benchmark.

## Key files

- **finetune_utils.py** — Large utility file. Key functions: `parse_particle_and_score()` (parses model text output into particle + score), `formatting_texts_func_single_seq()` / `formatting_texts_func_edit_pairs()` (format data for model input), `load_test_fn_from_file()` (loads test function params from JSON), `get_ehrlich_metrics_for_outputs()` (computes alignment metrics). Also has GPU memory management and checkpoint validation helpers.
- **finetune_ehrlich.py** — Ehrlich-specific fine-tuning tools.
- **ehrlich_fn_difficulty.py** — Analyzes difficulty/hardness of Ehrlich function instances.

## Test function

The Ehrlich function (`holo.test_functions.closed_form.Ehrlich`) scores discrete integer sequences based on how well they match hidden motif patterns. Particles are integer lists of length `dim`, with values in `[0, num_states)`. The optimization goal is to find sequences that maximize the score (minimize the negative score).
