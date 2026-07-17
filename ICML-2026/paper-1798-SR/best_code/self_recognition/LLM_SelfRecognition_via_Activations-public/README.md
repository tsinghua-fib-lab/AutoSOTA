# LLM Self-Recognition: Steering and Retrieving Activation Signatures

This repository contains the **evaluation pipeline** used in the paper _“LLM Self-Recognition: Steering and Retrieving Activation Signatures.”_ It **only evaluates** whether a model can distinguish its **own generated texts** from **human-authored texts** using internal activations. For **steering to amplify activation signatures**, see the other part of this repository.

Scope
- Generates model completions for summarization prompts.
- Extracts activations from specified layers/tokens.
- Trains per-layer LDA classifiers and reports AUROC/accuracy.
- Optional baselines: perplexity and mass-mean (nearest centroid).

Quickstart
1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. (Optional) Set a Hugging Face token for gated models. Put it in a `.env` file as `HF_TOKEN`.

```bash
# .env
HF_TOKEN=your_token_here
```

3. Run the pipeline.

```bash
python main.py
```

Use a paper config:

```bash
python main.py --config configs/paper_experiments/default_llama3-1b.yaml
```

Configuration
- `main.py` reads a YAML config. If `configs/default.yaml` does not exist, it is created automatically with defaults.
- Config sections: `dataset`, `generation`, `extraction`, `training`, `paths`.
- Key controls include dataset name/subset, model name, number of samples, layers/tokens to analyze, and classifier options.

Outputs
- Run folders are created under `evaluation_data/<dataset>/<model>/<run_name>/`.
- `config.json` stores the flattened config used for the run.
- `evaluation_results.json` stores per-layer metrics, best layer, and bootstrap CIs.

Notes
- Datasets are loaded via `datasets` and streamed from Hugging Face (`xsum`, `xl-sum`, or `c4`).
- Models are loaded via `transformers`; GPU is used automatically if available.
- `requirements.txt` pins a CUDA-enabled PyTorch build; install a CPU-only build if you do not have CUDA.
- This codebase only covers **self-recognition evaluation** and does not include the full steering/retrieval experiments.
