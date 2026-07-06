# Reproduction Notes for Paper 81: MAA (Robust Vision-Language Models via Manifold-Adversarial Adapters)

## Environment
- Container: autosota_repro_paper_81
- Base image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime (upgraded PyTorch)
- Python: 3.10.13
- PyTorch: 2.9.1+cu128
- CUDA: 12.8
- GPUs: 2x NVIDIA A100-SXM4-80GB (used GPU 0 only for inference)

## Dependencies
All dependencies from requirements.txt and VLMEvalKit requirements installed successfully.
- maa package installed in development mode (pip install -e .)
- vlmeval package installed from eval/VLMEvalKit
- Key packages: torch==2.9.1, transformers==4.46.3, tokenizers==0.20.3

## Bug Fix Applied
**File:** `/repo/maa/checkpoint.py` (function `load_maa_adapter_state`)
**Issue:** After loading MAA adapter weights with `model.load_state_dict()`, adapter parameters remained on CPU while the parent encoder layers were on CUDA (due to `low_cpu_mem_usage=True` device_map auto-splitting). This caused CUDA device mismatch errors during inference.
**Fix:** Added code to iterate through encoder layers and move each layer's `maa_adapter` sub-module to the same device as the layer's non-adapter parameters using `adapter.to(target_device)`.

## Model Weights
- Base model: liuhaotian/llava-v1.6-mistral-7b (downloaded via git-lfs from HF mirror, 29GB)
- MAA adapter: acnul/maa/maa.pth (97MB, downloaded from HF)

## Evaluation
- R-Bench-Dis: 495 questions, inference completed successfully (~7 min at ~1.2 it/s)
- R-Bench-Ref: Not yet run

## Blocking Issue: GPT-4.1 Judge Required
The paper uses GPT-4.1 as judge for R-Bench evaluation (`judge=GPT-4.1` in rubric).
Without an OpenAI API key, VLMEvalKit falls back to exact matching.
LLaVA models generate verbose descriptive text rather than concise answer letters,
making exact matching ineffective (produces 0% accuracy).

The `can_infer` evaluation function requires the answer letter to appear
within the last 5 words of the output, or the output text to be short enough
to match option text. LLaVA's verbose responses fail both checks.
