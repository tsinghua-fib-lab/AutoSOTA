# DMANet Code Analysis for SOTA Optimization

## Evaluation Path
- Entry: /repo/eval_etth1.py
- Flow: eval_etth1.py -> run.py (subprocess per pred_len) -> Exp_Long_Term_Forecast.train() -> test()
- Metrics: Parsed from stdout with regex mse:(NUMBER).*mae:(NUMBER)
- Output: /repo/output/eval_results.json (per-horizon + averaged)
- Eval timeout: 10 minutes per full run (4 pred_lens)

## Train/Inference Path
- Training: exp/exp_long_term_forecasting.py (Exp_Long_Term_Forecast class)
  - Optimizer: Adam (line 42) with no weight_decay
  - Loss: Frequency-domain MAE (auxi_loss=MAE, auxi_mode=fft, auxi_type=complex)
  - Scheduler: adjust_learning_rate() in utils/tools.py (type1 = halving each epoch)
  - Early stopping: patience=3
  - No gradient clipping
- Model: models/DMANet.py

## Config Path
- CLI args: run.py argparse
- Override: eval_etth1.py BASE_ARGS

## Known Bugs
1. DMANet.py line 57: x_out = self.act(x_out) should be x_out = self.drop(x_out)
   - self.drop = nn.Dropout(configs.dropout) created at line 30 but never called
   - self.act (GELU) applied twice instead of GELU -> Dropout

## Safe Modification Targets
1. models/DMANet.py - Model architecture
2. exp/exp_long_term_forecasting.py - Training loop
3. eval_etth1.py - Eval orchestration params
4. utils/tools.py - LR schedulers
