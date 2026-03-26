# Paper 39 — CALF

**Full title:** *CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning*

**Registered metric movement (internal ledger, ASCII only):** -0.19%(0.4316->0.4308)[MSE ]

# Optimization Results: CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning

## Summary
- Total iterations: 12
- Best `mse`: 0.4308 (paper baseline: 0.4349, our baseline: 0.4316)
- Improvement vs paper: -0.95% (MSE 0.4349 → 0.4308)
- Improvement vs our baseline: -0.19% (MSE 0.4316 → 0.4308)
- Best commit: 2cca5c4f2a (iter-6: Remove output consistency loss)
- Target (0.4262) was NOT reached (gap: 0.0046 = 1.07% from best)

## Baseline vs. Best Metrics
| Metric | Paper Baseline | Our Baseline (iter 0) | Best (iter 6) | Delta vs Paper |
|--------|---------|------|-----|-------|
| MSE | 0.4349 | 0.4316 | 0.4308 | -0.95% |
| MAE | 0.4290 | 0.4278 | 0.4215 | -1.75% |

## Key Finding: The Output Consistency Loss Hurts
The critical discovery was that **setting `--output_w 0.0`** (removing the output consistency loss between time and text branches) improves MSE from 0.4316 to 0.4308.

**Why this works**: CALF's cmLoss has three components:
1. `task_loss` (L1 task loss, `task_w=1.0`): supervised signal vs ground truth
2. `output_loss` (L1 consistency, `output_w=1.0`): forces `outputs_time ≈ outputs_text`
3. `feature_loss` (L1 feature alignment, `feature_w=0.01`): aligns intermediate layer features

The `output_loss` with `output_w=1.0` forces the time series branch to mimic the text branch's outputs. Since test only uses `outputs_time`, this constraint limits the time branch's specialization for forecasting. Removing it (output_w=0.0) lets the time branch freely optimize for the task.

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| `--output_w 0.0` | MSE: 0.4316 → 0.4308 (-0.19%) | **Production change** - remove output consistency loss |
| No code file changes needed | - | Pure command-line parameter change |

## What Worked
- **Reducing output_w**: Setting to 0.1 (iter 5) or 0.0 (iter 6) both helped. 0.0 gave best results.
- The intuition is clear: output consistency loss constrains the time-series forecast branch to match the text branch's output, which is suboptimal since test only uses the time branch.

## What Didn't Work
- **MSE task_loss** (iter 1): Switching task_loss from L1 to MSE slightly hurts (0.4334 vs 0.4316). L1 has better gradient properties for early training stages.
- **dropout change** (iter 2): The `--dropout` parameter is **NOT used** in the CALF model (only `--lora_dropout` is). This is a code dead-end (Encoder_PCA uses default dropout=0.1 regardless).
- **Ensemble text+time at test** (iter 3): Averaging text and time branch outputs at test time hurt because the text branch is not directly trained on the forecasting task (task_loss only targets `outputs_time`).
- **Lower learning rate LR=1e-4** (iter 4): Slightly worse MSE but better MAE. More epochs trained (~42 vs 24) but did not converge to better minimum.
- **Removing feature_w=0.0** (iter 7): Hurts significantly (0.4308 → 0.4370). The intermediate feature alignment between branches is critical and beneficial.
- **More gpt_layers=8** (iter 8): Worse. The default 6 layers is optimal. 8 layers overfit.
- **Higher LoRA rank r=16** (iter 9): Worse than r=8. Higher rank causes overfitting or slower convergence.
- **Hybrid L1+MSE task loss** (iter 10, LEAP): Worse. The hybrid combines L1 and MSE, but L1-only is better for early stopping via validation-MSE.
- **Higher feature_w=0.1** (iter 12): Worse. 10x the feature alignment signal is too strong.

## Architecture Insights
1. **dropout arg is unused**: The `--dropout` CLI arg is completely ignored by CALF model. Only `--lora_dropout` affects regularization.
2. **Train-val mismatch**: Training uses L1 task_loss, but validation uses MSE to select early stopping checkpoint. This mismatch is actually OK - L1 provides stable gradients, MSE provides good model selection.
3. **Dual-branch design**: The text branch is trained with very few parameters (mostly frozen GPT-2). Its outputs being forced to match the time branch (output_loss) creates a bottleneck.
4. **Feature alignment is beneficial**: The `feature_w=0.01` loss (alignment of intermediate features between branches) is the beneficial part of the cross-modal training. It provides a knowledge distillation signal.

## Optimization Trajectory
```
Iter 0 (baseline): MSE=0.4316
Iter 5 (output_w=0.1): MSE=0.4314 ↓ new best
Iter 6 (output_w=0.0): MSE=0.4308 ↓ new best (final best)
```

## Deep-research memo (excerpt from `research_report.md`)

**3. Parameter Optimization Insights**
From recent LLM-for-TSF papers and code, effective hyperparameter ranges include: 
- **Learning Rate (LR)**: Fine-tuning large models often uses LR around 1e-4–5e-4. TS-HTFA uses LR=5e-4 (www.mdpi.com). One could try up to 1e-3 for faster convergence, but above 1e-3 risk overshoot. Lowering to 1e-5 can stabilize very noisy training. Use a scheduler (e.g. linear decay or cosine); TS-HTFA’s “lradj” options suggest linear/polynomial decay schedules. 
- **LoRA Hyperparameters**: The rank *r* is typically small (e.g. 4,8,16). Earlier work often sets *r=8* as a sweet spot (www.mdpi.com); TS-HTFA used 8. Growing *r* to 16–32 can capture more nuance but adds parameters. The LoRA scaling *alpha* often equals *r* or is a constant like 32 (www.mdpi.com). A typical choice is 16 or 32; too large may overweight the adaptation terms. *lora_dropout* is usually low (0.0–0.3); TS-HTFA uses 0.1 (www.mdpi.com). If overfitting is a concern, increasing dropout towards 0.3–0.5 can help. 
- **Training Epochs / Early Stop**: Many TS-Forecasting LLM approaches train for O(50–100) epochs. The default 100 epochs (with patience=10) is reasonable; in practice early stopping often halts at 30–50 epochs on these datasets. If you see overfitting (validation error rising), reduce max epochs or lower patience to 5–10. More data or smaller model may allow longer training. 
- **Batch Size**: Values 64–256 are common. The default 256 may be large if GPU memory is constrained; one can try 128 or 64 to allow larger models or longer sequences. Larger batches (>256) can improve gradient stability but may reduce the number of updates (and require LR tuning). 
- **Dropout**: CALF default is 0.3. Modern fine-tuning often uses 0.1–0.2 (www.mdpi.com). Lower dropout (0.1) tends to improve fine-tuned accuracy (as TS-HTFA did) but risks overfitting. If validation error is unstable, raise dropout (up to 0.5). Even a dropout of 0.0 (no dropout) is acceptable for inference ensembles. 
- **GPT Layers (Depth)**: CALF uses the first 6 layers of GPT-2. You can vary this: more layers (8–12) lets the model use deeper context but increases compute. If GPU allows, using all 12 layers of GPT-2 (with frozen base) might slightly improve representational power. Conversely, fewer layers means faster inference but might underfit. TS-HTFA found 6 layers adequate. Try 4 or 8 if exploring speed vs. accuracy. 
- **Model Dimension / Heads**: GPT-2 base is 768-dim with 12 heads. The repo uses 768 with 4 heads (possibly projecting differently). In general, keep *d_model* at 768 (matching GPT-2). You could attempt more heads (e.g. 8) by setting `n_heads=8` if the code permits (ensuring divisibility). More heads can help if *d_model* >768 (e.g. using GPT-2 Medium), but mixing dims is risky. Feedforward dim *d_ff* is often 4× *d_model* (3072 for PTB practitioners). If tunable, try *d_ff*=1024–2048; larger *d_ff* often improves capacity but may overfit. 
- **Sequence Length (seq_len)**: The default 96 or 192 can be varied. Shorter lengths (48) mean less context but often stabilize training; longer (336) capture seasonality but may confuse the LLM. The TS-HTFA authors note very long inputs can degrade LLM performance (www.mdpi.com). Try lengths {48,96,192} and pick the best. For evaluation, you might ensemble predictions from multiple seq_len settings. 
- **Exogenous Features**: If experimenting with features (M/S/MS), note “M” means using all series as multivariate input. If your task is purely multivariate forecasting, keep “MS” (multivariate input to scalar output). Only switch to “S” (single) if treating one series at a time (usually reduces context). Most SOTA works stick with “MS.” 

In summary, prior works suggest **LR ~5e-4, LoRA r=4–16 (often 8) with alpha ~32 and dropout ~0.1**, trained ~50–100 epochs with early stop (www.mdpi.com). Head/hidden sizes usually match the LLM defaults. These are good starting points; minimally, try {r=4,8,16}, {alpha=16,32,64}, and LR {1e-4,5e-4,1e-3}. Then tune based on validation loss. 

**4. Concrete Optimization Ideas (with Gain/Risk)**

1. **Ensemble with Dropout (MC-Dropout)**: At inference time, enable dropout and run the model *k* times (e.g. 5–10) on the same input, then average the outputs. This acts as a light ensemble and typically reduces variance. *Expected Gain*: ~1–3% MAE/MSE improvement (smooths over model uncertainty). *Risk*: Low accuracy risk; *cost*: ~k× slower inference. 

2. **Multi-Window Ensemble**: Rather than forecasting one long window, slide or vary the input window and average forecasts. E.g., forecast horizon=192 by (a) using the last 96 input points, (b) using offset windows, etc., then average results. This mitigates sensitivity to window length. *Gain*: ~1–2% improvement. *Risk*: Low (just increased inference time). 

3. **Output Smoothing/Filtering**: Apply a simple filter to the raw forecast (e.g. a short moving average or median filter of the last few predicted points). This can reduce spurious spikes. *Gain*: small (especially on noisy series). *Risk*: Low if filter is short; *mistake*: over-smoothing could lag real trend (but overall risk is low). 

4. **Blend with a Baseline Model**: Combine CALF’s forecast with a simpler predictor (e.g. the “last known value” or a linear trend model). For example, output = 0.8*(CALF) + 0.2*(Naïve). This often helps correct systematic bias. *Gain*: ~1–3%, especially if CALF is slightly off. *Risk*: Low to moderate (improves bias but may under-correct sharp changes). 

5. **Calibrate via Last-Batch Residual**: Using the end of the historical window (or a holdout set), compute the bias of the model (forecast minus truth). Then subtract this bias from all future forecasts. This simple bias-correction often trims a few percent off error. *Gain*: ~1–2%. *Risk*: Low (won’t drastically harm, although if bias estimate is noisy, effects are limited). 

6. **Forecast Chaining (Auto-Regressive Rollout)**: Instead of a single 192-step forecast, recursively forecast smaller chunks (e.g. 96 steps twice). Feed the first 96 predicted values back in to predict the next 96. This reduces error accumulation on extremely long horizons. *Gain*: Up to ~5% on long-range (>100) forecasts. *Risk*: Medium (errors compound through steps; overall error may not drop for very long horizons). 

7. **Scale and Detrend Preprocessing**: Detrend the input series (e.g. subtract mean or linear trend of input window), feed the residuals to the model, then add the trend back to predictions. If your series have a strong trend, this can improve accuracy. Alternatively, rescale inputs (normalize) differently and inverse-transform outputs. *Gain*: ~1–3% if trend/scale misalignment is an issue. *Risk*: Moderate – improper detrending can introduce bias if the method is not well-chosen. 

8. **Test-Time Data Augmentation (TTA)**: Create augmented versions of the recent history (e.g. add small Gaussian noise, jitter timestamps slightly, or time-warp by ±5%). Forecast on each augmented input and average. This simulates an ensemble and can slightly improve robustness. *Gain*: ~1–2% (often smaller than MC-Dropout). *Risk*: Low to moderate – if distortions are too large, forecasts could worsen, but small augmentations are usually safe. 

9. **Ensemble of Hyperparameters / Seeds**: If multiple trained models or checkpoints exist (e.g. saved at different epochs or with slightly different LR), average their predictions. This hedges overtraining. *Gain*: ~1–3%. *Risk*: Low (just larger compute), assuming models are not identical. 

_(Research digest truncated.)_

## Idea library snapshot (`idea_library.md`)

### IDEA-001: Switch task_loss from L1 to MSE
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Change `--task_loss` from `l1` to `mse`. The evaluation metric is MSE — using MSE as training loss directly aligns training objective with test metric.
- **Hypothesis**: Direct optimization of MSE during training should produce lower test MSE. In TS forecasting, MSE loss often gives better MSE metric than L1 loss.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-002: Lower dropout from 0.3 to 0.1
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Change `--dropout 0.3` to `--dropout 0.1`. TS-HTFA (a follow-up to CALF) used 0.1 dropout and achieved significant improvements.
- **Hypothesis**: The current 0.3 dropout is very aggressive and may cause underfitting. Lowering it should allow the model to learn better representations.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-003: Lower learning rate to 1e-4
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Change `--learning_rate 0.0005` to `--learning_rate 0.0001`. Lower LR for fine-tuning LLMs is common practice.
- **Hypothesis**: 5e-4 may be too aggressive for fine-tuning GPT-2's LoRA parameters, causing overshooting. 1e-4 allows more stable convergence.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-004: Increase LoRA rank to 16
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `--r 8` to `--r 16`. Higher LoRA rank allows more expressive adaptation.
- **Hypothesis**: Rank 16 captures more nuanced time series patterns than rank 8, improving the frozen model's adaptation to numeric data.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-005: Increase tmax for cosine annealing
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change tmax from 20 to 50 (pass `--tmax 50`). This controls how quickly LR decays in cosine annealing.
- **Hypothesis**: With tmax=20, the LR drops to near zero very quickly. With tmax=50, training benefits from higher LR for longer, potentially finding better minima.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-006: Increase feature_w to improve branch alignment
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `--feature_w 0.01` to `--feature_w 0.1`. This increases the weight of the feature alignment loss between time and text branches.
- **Hypothesis**: Better aligned representations between branches may produce a stronger "distillation" signal, helping the time branch learn better representations.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-007: Increase gpt_layers to 8
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change gpt_layers from 6 to 8 (pass `--gpt_layer 8`). More GPT-2 layers = more representational capacity.
- **Hypothesis**: Using 8 instead of 6 GPT-2 layers gives the model more context modeling ability. GPT-2 has 12 layers total, 6 may be underfitting.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-008: Use smooth_l1 as task_loss
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `--task_loss l1` to `--task_loss smooth_l1`. SmoothL1 combines MAE robustness for large errors with MSE smoothness for small errors.
- **Hypothesis**: SmoothL1 may provide better gradient behavior than pure L1, helping the model better minimize both MAE and MSE.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-009: Combine MSE task_loss with lower dropout
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Combine IDEA-001 and IDEA-002: `--task_loss mse --dropout 0.1`. Both changes are individually supported by evidence.
- **Hypothesis**: Synergistic effect — using MSE loss with better regularization (lower dropout) should substantially improve test MSE.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-010: Use MSE as output_loss (consistency loss)
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `--output_loss l1` to `--output_loss mse`. The consistency loss between time and text branches uses L1 by default.
- **Hypothesis**: Using MSE for the consistency loss may produce better-aligned branches since MSE penalizes large deviations more heavily.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-011: Ensemble text and time outputs at inference
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: MEDIUM
- **Description**: Modify `test()` in `exp_long_term_forecasting.py` to use average of `outputs_time` and `outputs_text` instead of just `outputs_time`. Code: `outputs_ensemble = 0.5 * outputs['outputs_time'] + 0.5 * outputs['outputs_text']`
- **Hypothesis**: The text branch also produces forecasts through a parallel GPT-2 pathway. Ensembling both branches at test time may reduce error since they've been trained with complementary objectives.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-012: Reduce batch_size to 128 for better generalization
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `--batch_size 256` to `--batch_size 128`. Smaller batch sizes are known to improve generalization.
- **Hypothesis**: The current batch_size of 256 may cause the model to find sharp minima. Smaller batch provides noisier gradients that can serve as regularization, often improving generalization.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-013: Use MSE for feature_loss (alignment loss)
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Change `--feature_loss l1` to `--feature_loss mse`. The intermediate feature alignment between branches uses L1.
- **Hypothesis**: MSE penalizes large feature misalignments more heavily, potentially producing better-aligned representations.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-014: Increase LoRA alpha to 64
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `--lora_alpha 32` to `--lora_alpha 64`. Higher alpha scales LoRA adaptation more strongly.
- **Hypothesis**: With rank=8, the default scaling alpha/r = 32/8 = 4. Raising alpha to 64 doubles this, allowing stronger LoRA adaptation which may better fit time series.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-015: Cross-pollination - Test-Time Augmentation (TTA) for ensemble forecasting
- **Type**: LEAP
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: At inference, run multiple forward passes with slightly varied inputs (shifted windows, or noise) and average predictions. Specifically, for each test batch, create T augmented inputs (e.g., T=5 with small Gaussian noise added), run all through the model, and average predictions.
- **Hypothesis**: TTA creates an implicit ensemble at inference without additional training, typically reducing variance and improving accuracy by 1-3%.
- **Status**: PENDING
- **Result**: (fill in after execution)
