# Code Analysis - Paper 5155

## Evaluation path
- `eval_slam.py` → loads checkpoints → evaluates RMSE + cosine similarity
- Uses `FlowMatching.sample_ode()` with Euler integration
- Decodes via nearest grid point with `grid_resolution=128`

## Train/inference path
- Training: `utils/training.py:TrainingManager.train()` → `FlowTrainer` or `FeedforwardTrainer`
- Model: `cleanup_ssps/model.py:ResidualMLP` (3-block ResMLP, GELU, time_embed_dim=32, dropout=0.1)
- Flow matching: `cleanup_ssps/cleanup_methods.py:FlowMatching`
- Config: `configs/config.yaml`

## Metric parser
- Parses stdout: `RMSE: <value>` and `Cosine: <value>` lines

## Safe modification targets
1. `eval_slam.py`: grid_resolution (128→256), num-steps CLI arg
2. `cleanup_ssps/cleanup_methods.py`: sample_ode() integration method, sigma_min
3. `cleanup_ssps/model.py`: time_embed_dim, block count, dropout
4. `cleanup_ssps/run.py`: loss function, LR scheduler, sigma_min schedule
5. `utils/training.py`: _ot_for_mode() OT pairing policies
6. `configs/config.yaml`: epochs, sigma_min, lr, sampling_modes

## Risky files (do not modify)
- `utils/evaluation_utils.py`: metric computation
- `cleanup_ssps/dataset.py`: test data generation
- `cleanup_ssps/sspspace.py`: SSP space (affects metric computation)
- `eval_slam.py` metric definition (rmse/cosine computation)

## Key findings
- geo_det uses random coupling (no OT) in `_ot_for_mode()`
- geo_amb_sb uses Sinkhorn OT with SB correction - fully implemented, just needs training
- ODE integration is simple Euler (first-order) - upgradeable to midpoint/RK4
- CosineEmbeddingLoss only optimizes direction, not magnitude
- 100 epochs at batch 256 with 18K train samples = ~7K optimization steps
- Grid resolution 128 → 256 would halve quantization error
