## `activations.py`

This contains the code we use to pass an input into a LM and SAE and extract its activations using NNsight.

## `data.py`

This contains our data loading and preprocessing code for the experiments in the paper. It has a `load_data` function which loads datasets from HuggingFace, and then additional functions to pass the loaded data through a model/SAE to get a set of activations to use for analysis.

## `process_sae_bench_autointerp`

This snippet takes the output of SAE Bench's autointerpretability code and processes it into a dictionary of feature: explanation pairs. Simply call

```
process_sae_bench_autointerp.py --autointerp-path 'path/to/saebench/autointerp/result.json'
```