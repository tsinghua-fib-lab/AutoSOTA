# Code Analysis for Paper 517: HO-SFL

## Evaluation Path
- Entry: `/repo/HO_SFL_LLM/main_ho_sfl.py`
- Runner: `/repo/HO_SFL_LLM/src/runner/ho_sfl_runner.py`
- Evaluation in `HO_SFLRunner.evaluate()` (lines 214-238): iterates test_loader, computes accuracy via `last_token_accuracy`
- Output format: `[Eval] Round N | Validation Accuracy: X.XXXX`
- Parsing: grep for `[Eval] Round` lines, highest value across all rounds

## Train/Inference Path
- `HO_SFLRunner.run()`: Main training loop
  - Samples clients each round
  - ZO gradient estimation via `HybridGradientEstimator.compute_v_with_server()`
  - Server optimizer step (AdamW)
  - Client gradient update via `update_from_aggregated_v()`
  - Client optimizer step (AdamW)
- Split model: client (layers 0..split_point-1) + server (split_point..last)
- LoRA applied to both client and server: r=8, alpha=16, target_modules=["q_proj", "v_proj"]

## Config Path
- `/repo/HO_SFL_LLM/src/utils/cli_parser.py`: All CLI settings
- `ModelSetting`: model name, dtype, LoRA r/alpha, split_point
- `OptimizerSetting`: optimizer, lr, momentum, beta1, beta2
- `HybridGradientEstimatorSetting`: mu, num_pert
- `HO_SplitFederatedLearningSetting`: num_clients, sampled_client_num, total_steps, evaluation_interval
- `DataSetting`: dataset, max_length, batch sizes, iid flag

## Metric Parser
- `prepare_settings.py:get_metrics()`: Creates MetricPacks with train_loss, test_acc
- `language_utils.py:last_token_cross_entropy_loss()`: Cross-entropy on last token logits
- `language_utils.py:last_token_accuracy()`: Accuracy from last token prediction vs label

## Risky Files
- `hybrid_gradient_estimator.py`: Core ZO math — changes here affect gradient quality
- `ho_sfl_runner.py`: Training loop — changes affect convergence
- `opt_split.py`: Already patched for NaN fix in attention

## Safe Modification Targets
- `hybrid_gradient_estimator.py:_generate_noise_and_apply()`: Noise distribution (Rademacher)
- `hybrid_gradient_estimator.py:update_from_aggregated_v()`: Gradient clipping, EMA
- `ho_sfl_runner.py:run()`: Scheduler calls, mu annealing, adaptive P
- `prepare_settings.py:get_optimizer()`: Scheduler addition
- `cli_parser.py`: New CLI arguments
- `language_utils.py:last_token_cross_entropy_loss()`: Label smoothing

## Reusable Resources
- `/autosota_cache/hf`: HuggingFace model cache
- `/datasets`: Dataset cache (SST-2 from GLUE)
- No /paper_data mount

## Baseline
- Accuracy: 87.73% (iteration 0)
- Command: main_ho_sfl.py --seed=42 --cuda --device=cuda:0 --large-model=opt-125m --lora --lora-r=8 --lora-alpha=16 --split --split-point=3 --optimizer=adamw --lr=1e-5 --mu=1e-3 --num-pert=2 --num-clients=10 --sampled-client-num=3 --total-steps=2000 --evaluation-interval=25 --dataset=sst2 --max-length=128 --train-batch-size=32 --test-batch-size=200 --iid --total-samples=80000
