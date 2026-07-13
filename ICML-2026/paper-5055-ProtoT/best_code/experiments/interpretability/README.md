# Interpretability Analysis Code

Scripts for analysing and comparing the interpretability of **ProtoAttn prototype features**, **SAE features** from a baseline LLaMA, and **LLaMA attention heads** via value-norm analysis.

---

| Script | Purpose |
|---|---|
| `find_proto_activations.py` | Extracts per-prototype routing weights (Pi) from a ProtoAttn model; saves the top-k activating sentences per (layer, prototype) as HTML + JSON. |
| `create_null_activations.py` | Builds a null baseline by sampling sentences uniformly at random across prototypes; used as a control in the LLM-scoring step. |
| `run_LLM_scoring.py` | Calls an OpenAI chat model to score thematic disentanglement of each prototype or SAE feature (`disentanglement_score`, `coverage_main_theme`, `number_of_themes`). |
| `train_SAE_LLaMA.py` | Trains TopK Sparse Autoencoders on the post-attention residual stream of LLaMA, one per layer. |
| `find_SAE_activations.py` | Extracts the top-activating sentences per SAE feature; writes a JSON with the same schema as the ProtoAttn output for direct comparison. |
| `find_llama_activations.py` | Computes per-head value-vector norms across LLaMA attention heads; outputs an HTML heatmap and per-head sparsity metrics (Gini, entropy, mutual information, …). |

---

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers tokenizers tqdm numpy scikit-learn
pip install openai                          # for run_LLM_scoring.py
```

`train_SAE_LLaMA.py` and `find_SAE_activations.py` also require the [`dictionary_learning`](https://github.com/saprmarks/dictionary_learning) library:

```bash
git clone https://github.com/saprmarks/dictionary_learning
pip install -e dictionary_learning
```

The scripts also import `data_utils.py` (all scripts) and `prototype_attn.py` (`find_proto_activations.py`) from the parent repository — make sure these are on your `PYTHONPATH` or in the same directory.

---

## Usage

### ProtoAttn activations

Edit `LOAD_DIR` at the top of `find_proto_activations.py`, then:

```bash
python find_proto_activations.py
```

Outputs an HTML heatmap and a JSON file under `prototype_analysis_word_level_<MODEL_ABL>/`.

### Null baseline

Update `INPUT_JSON_PATH` in `create_null_activations.py` to point at the JSON above, then:

```bash
python create_null_activations.py
```

### LLM scoring

Set `INPUT_JSON_PATH` in `run_LLM_scoring.py` (ProtoAttn JSON, null JSON, or SAE JSON), then:

```bash
export OPENAI_API_KEY="sk-..."
python run_LLM_scoring.py
```

### SAE training

```bash
python train_SAE_LLaMA.py \
    --model_path /path/to/llama_hf_checkpoint \
    --fineweb    data/FineWeb/train.npz \
    --save_root  ./sae_outputs \
    --seeds 0 1 2
```

See `--help` for the full argument list (dict size, sparsity k, learning rate, layer range, …).

### SAE activations

```bash
python find_SAE_activations.py \
    --model_path     /path/to/llama_hf_checkpoint \
    --sae_root       ./sae_outputs \
    --tokenizer_path tok/fineweb_bpe_16000.json \
    --val_path       data/FineWeb/val.npz \
    --output_dir     ./sae_analysis \
    --seeds 0 1 2
```

### LLaMA attention head analysis

Edit `MODEL_DIR`, `FINEWEB_VAL`, `OUTPUT_DIR`, and `METRICS_DIR` at the top of `find_llama_activations.py`, then:

```bash
python find_llama_activations.py
```

---
