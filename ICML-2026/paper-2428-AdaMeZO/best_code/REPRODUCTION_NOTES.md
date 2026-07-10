# AdaMeZO Reproduction Notes

## Paper
- Title: AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments
- Target: RTE, RoBERTa-large (350M), K=16 few-shot, Prompt-based fine-tuning
- Paper accuracy: 63.1% ± 2.3 (AdaMeZO), 61.6% ± 1.3 (MeZO)

## Reproduction Results
- AdaMeZO (hw=10, beta1=0.7, beta2=0.9), max_steps=2000
  - Seed 13: 49.8%
  - Seed 21: 60.3%
  - Seed 42: 50.9%
  - Seed 87: 53.1%
  - Mean ± Std: 53.5% ± 4.7%
- MeZO (hw=0), max_steps=2000
  - Seed 21: 55.6%

## Key Changes from Paper
1. max_steps=2000 instead of 100000 (model diverges past ~1700 steps with ZO)
2. Best-model checkpointing added to trainer (saves at best eval loss, loads for final eval)
3. Early stopping tolerance increased from 5 to 500 (paper uses patience=5 at eval_steps=10000)
4. RTE data from HuggingFace GLUE instead of Princeton datasets.tar

## Dependencies
- PyTorch 2.1.0, CUDA 12.1
- transformers==4.28.1
- datasets, scikit-learn, accelerate, sentencepiece
- loralib, wandb (disabled), matplotlib, seaborn
- jq (apt), wget (apt)

## Eval Command
```bash
bash /repo/eval_adaMeZO_rte.sh
```

## Patches Applied
See /repo/MeZO/medium_models/src/trainer.py:
1. Line 563: tolerance > 5 -> tolerance > 500
2. Line 548-549: best_eval_loss and best_model_state_dict initialization
3. Line 739-740: save best model state on eval improvement
4. Line 1015-1017: load best model before final eval
