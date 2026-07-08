# Code Analysis for CDKA SOTA Optimization (Paper 1418)

## Evaluation Path
- Entry: run_exp.py -> train_text_to_text_model() -> trainer.train() -> trainer.evaluate()
- Metric: utils.py:compute_metrics() - soft accuracy on first generated token
- Metric Parsing: Log line FINAL_EVAL_ACCURACY in utils.py:train_text_to_text_model()
- Data: MNLI from GLUE, loaded via data.py:load_mnli(), mapped to entailment/neutral/contradiction

## Training Path
- Trainer: logTrainer.py:LogTrainer (extends HF Seq2SeqTrainer)
- Model init: initialize_text_to_text_model() -> T5-Base via AutoModelForSeq2SeqLM
- PEFT: Custom peft/src/peft/tuners/lora/ with Kronecker adapter (CDKA)
- LoraConfig: Uses r1, r2, r (Kronecker decomposition ranks), lora_alpha, use_rslora

## Config Path
- Model: conf/model/t5base.yaml - epochs=1, bs=32, real_bs=32, bf16=False, lr=2e-3
- PEFT: conf/peft/all.yaml - r1=null, r2=null, r=null, alpha=64 (CLI overrides to 16)
- Init: conf/init/default.yaml - mode=simple, lora_A=kaiming, lora_B=zeros

## Key Files
- run_exp.py - Main entry, reinit_lora_modules(), estimate_gradient(), kron()
- utils.py - train_text_to_text_model(), compute_metrics()
- logTrainer.py - LogTrainer with training_step()
- peft/src/peft/tuners/lora/layer.py - Linear.forward() with Kronecker, merge()
- peft/src/peft/tuners/lora/config.py - LoraConfig with r1, r2, r fields
- data.py - Dataset loading functions
- split.py - rebuild() for reshaping gradients

## CDKA Forward Pass (Kronecker Adapter)
1. Input x reshaped to (B, L, d_B, d_A) where d_A=in_features//r2, d_B=r2*r
2. lora_A: Linear(d_A, r1*r) applied
3. Reshaped+permuted through intermediate dimensions
4. lora_B: Linear(r2*r, out_features//r1) applied
5. Result scaled and added to base layer output
6. Effectively: kron(lora_B, lora_A) * scaling + base_weight

## Scaling
- With use_rslora=True: scaling = lora_alpha * (1 / (r*r2))^0.5
- Without rslora: scaling = 1

## Known Issues
1. compute_metrics() uses np.unique(label_ids) per batch - if batch lacks a class, that class cannot be correctly predicted
2. MNLI load_mnli() uses download_mode=force_redownload but HF mirror caches
3. lr_scheduler_type=cosine is hardcoded in utils.py

## Safe Modification Targets
- conf/model/t5base.yaml - epochs, lr, bf16, batch sizes
- conf/peft/all.yaml - r1, r2, r, alpha
- conf/init/svd.yaml, conf/init/gradient.yaml - init configs
- run_exp.py:reinit_lora_modules() - SVD init mode
- utils.py:compute_metrics() - fix batch-dependent unique_labels
- logTrainer.py:training_step() - MGPO perturbation
- peft/src/peft/tuners/lora/layer.py - DoRA magnitude, merge

## Unsafe Files (DO NOT MODIFY)
- data.py - Dataset loading/splits/labels
- hf_patch.py, patch_*.py - HF compatibility patches
- /tools/record_score.sh - Scoring script
