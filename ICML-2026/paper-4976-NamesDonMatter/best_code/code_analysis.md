# Code Analysis — Paper 4976 SOTA

## Evaluation path
- Entry: `autoregltl/main.py` -> `eval-ted` -> `autoregltl/eval.py:evaluate()` -> `evaluate_model()` -> `model.generate_predictions()` -> beam search -> trace checking
- Metrics parsed from stdout: `Correct: X/10000` and `Exact match: Y/10000` percentage lines
- Alpha-covariance from resym eval: `resym-eval-ted` without `--max-perm`

## Train/inference path
- Training: `autoregltl/main.py` -> `train-ted` -> `autoregltl/train.py:train()` -> HuggingFace Trainer
- TED trainer: `autoregltl/ted/train.py:TedTrainer` extends `LTLTrainer`
- Model: `autoregltl/ted/model.py:Transformer` (encoder-decoder with parallel AP streams)
- Loss: `autoregltl/losses.py:AdaCos` (adaptive cosine scaling)
- Inference: `Transformer.generate()` -> `BeamSearch.search()` with KV cache

## Baseline config
- d_embed_enc=96, d_embed_dec=96, d_ff=768, num_heads=6, num_layers=6
- Per-head dim = 16 (potential low-rank bottleneck)
- dropout=0.1, ff_activation=relu, enc_pe=sinusoid, dec_pe=rope
- cross_attn=per, tree_pos_enc=true
- loss=adacos, lr=1e-3, cosine schedule, warmup=1000
- epochs=64, batch_size=256, grad_acc=4 -> effective batch=1024
- train_max_samples=80000 -> ~78 steps/epoch -> ~5000 total steps
- eval_steps=3000

## Safe modification targets
- Beam search: alpha, beam_size (inference-only, fast)
- Training: epochs, LR schedule, loss function
- Architecture: d_embed, num_heads, num_layers, dropout
- Training fix: gradient accumulation normalization

## Training time estimate
- Baseline: 64 epochs x 78 steps/epoch ≈ 5000 steps
- On 2xL40S GPUs: ~2-3 hours for full training
