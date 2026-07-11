# Code Analysis — Hide&Seek Syn4 (Paper 3942)

## Evaluation Path
- `python3 utils/run_syn4_eval.py` → loops 20 seeds, calls `run_feature_selection_model()`
- Each seed: generates Syn4 data (11 features, 10K train/10K test), trains hide_and_seek model, computes TPR/FDR
- TPR/FDR computed per test instance via `performance_metric()`: element-wise comparison of binary mask (m > 0.5) vs ground truth
- Final output: median TPR, median FDR over 20 seeds

## Training Path
- `tools.run_feature_selection_model()` → `create_data()` → `generate_data()` → `Label_Generation()`
- Training: `model.train_nn()` in `hide_and_seek/model.py`
  - Full batch training (batch_size=None)
  - Adam(lr=0.001), 500 epochs
  - Loss: CE(y_pred, y_true) + lambda * (epoch/N)^lambda_exponent * mean(mask)
  - Perturbation: draw_marginal (independent column shuffle)
- Prediction: `model.pred_nn()` → generates perturbed samples from train set, runs forward pass

## Config Path
- Hard-coded in `utils/run_syn4_eval.py`:
  - lmbda=0.3, epochs=500, num_syn_features=11, train_N=10000, test_N=10000
  - hide_hidden_dim=32, seek_hidden_dim=32
  - hide_num_hidden_layers=2, seek_num_hidden_layers=2
  - lmbda_exponent=2, perturbation_method=draw_marginal

## Metric Parser
- stdout: "TPR mean: XX.X%" and "FDR mean: XX.X%"
- Final summary: "TPR: XX.XX", "FDR: XX.XX" (median over 20 seeds)

## Model Architecture
- hide_net: Linear(11→32) + ReLU + Linear(32→32) + ReLU + Linear(32→11) + Sigmoid → mask m in [0,1]
- seek_net: Linear(11→32) + ReLU + Linear(32→32) + ReLU + Linear(32→2) → logits
- Input to seek_net: m*x + (1-m)*x_hat where x_hat from perturbed training data

## Perturbation Methods
- draw_marginal: independent column-wise shuffle (current default)
- conditional_rf: RF-based conditional resampling per feature (preserves correlations)
- knock_off: DeepKnockoffs (heavy, requires training knockoff machine)

## Key Levers
1. lmbda (0.3): regularization strength — higher → sparser masks
2. lmbda_exponent (2): annealing schedule exponent
3. epochs (500): training duration
4. hide_hidden_dim/seek_hidden_dim (32): network capacity
5. hide_num_hidden_layers/seek_num_hidden_layers (2): network depth
6. perturbation_method: draw_marginal | conditional_rf | knock_off
7. mask threshold (0.5): binarization cutoff

## Safe Modification Targets
- hide_and_seek/model.py: train_nn() — optimizer, scheduler, architecture
- hide_and_seek/model.py: TemperatureScaledSigmoid — temperature parameter
- hide_and_seek/perturbation_methods.py: perturbation methods
- utils/run_syn4_eval.py: hyperparameters passed to run_feature_selection_model()

## Do NOT Modify
- utils/tools.py: performance_metric(), create_data()
- utils/Data_Generation.py: data generation, ground truth
- /tools/record_score.sh: scoring infrastructure
