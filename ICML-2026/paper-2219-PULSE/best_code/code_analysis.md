# PULSE Code Analysis for SOTA Optimization

## Evaluation Path
- evaluate_electricity.sh runs python -u run.py for each pred_len in {96, 192, 336, 720}
- Training: exp/exp_main.py:train() -> Exp_Main.train(setting)
- Testing: exp/exp_main.py:test() -> outputs mse/mae to result.txt
- Per-horizon hyperparameters differ (d_model, inv_len, patch_size, time_dim, dsa, ksize)

## Key Files
- models/PULSE.py: Model definition (backbone, phase router, ReVIN, MixUp)
- exp/exp_main.py: Training loop, loss computation, eval
- run.py: CLI argument parsing and entry point
- data_provider/data_loader.py: Data loading (Dataset_Custom for Electricity)
- evaluate_electricity.sh: Evaluation script

## Critical Finding: Training Loss
- With --rec_lambda 0 --auxi_lambda 1 (current config), ONLY FFT auxiliary loss is used
- The MSE loss block "if self.args.rec_lambda:" is skipped when rec_lambda=0
- Model trains purely on frequency-domain matching via complex FFT difference
- Adding --rec_lambda > 0 would add MSE supervision

## Model Architecture
- Backbone: nn.Linear(seq_len, 512) -> GELU -> nn.Linear(512, pred_len) (HARDCODED 512)
- Phase router: FullAttention with d_model from config (16 or 32)
- ReVIN normalization with learnable affine params
- MixUp augmentation with Beta(0.15, 0.15)
- FFT auxiliary loss: |fft(pred) - fft(true)|.abs().mean()

## Config Audit (all verified)
- seq_len=96, pred_len={96,192,336,720}
- lr=0.005, batch_size=64, epochs=30
- auxi_lambda=1, rec_lambda=0, seed=2024
- Data split: 7:1:2 (Dataset_Custom)
- global_period_W=168 (cycle_index uses mod 168)
- MixUp: Beta(0.15, 0.15) (manifest SAM alpha=0.15)

## Safe Modification Targets
- exp/exp_main.py loss computation: add rec_lambda, freq reweighting
- models/PULSE.py: attention modification, sparsity regularization
- run.py: add new CLI args if needed
- evaluate_electricity.sh: add new flags to evaluation runs
