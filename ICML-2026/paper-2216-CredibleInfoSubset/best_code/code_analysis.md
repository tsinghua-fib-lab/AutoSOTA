# Code Analysis - Paper 2216 (Credible Information Subset Decomposition)

## Evaluation Path
- **Entry point**: /repo/bandgap/train/eval_final.py
- **Execution**: cd /repo/bandgap && python3 train/eval_final.py
- **Training**: Inline in eval_final.py — trains from scratch, early-stops, evaluates on HSE test set
- **Output**: stdout prints MAE, RMSE, tau_b; saves to /repo/bandgap/reproduction_result.json

## Key Files
1. **/repo/bandgap/model/model_mine.py** — Main model definition:
   - BandModelSE: Top-level model (composition embedding -> VAE + rank encoder + interval encoder -> evidential predictor)
   - CondVAE: Conditional VAE with 64-dim latent, reconstructs composition + total features
   - RankAwareEncoder: Feature extraction -> fidelity conditioning -> rank attention -> rank features
   - IntervalAwareEncoder: Predicts tau (credible interval width) for NIG evidence scaling
   - EvidentialRegressor: NIG evidential regression head -> mu, v, alpha, beta
   - combined_vae_evidential_loss_SE: Multi-task loss (VAE + evidential + pair-wise ranking)
   - **Safe targets**: KL weight (line: 1e-3 * kld), loss weights (alpha_abs=1.0, alpha_rank=5e-3), activation functions

2. **/repo/bandgap/train/methodmine.py** — Training utilities:
   - train_model: Full training loop with early stopping
   - train_one_epoch: Single training epoch
   - eval_one_epoch: Validation epoch
   - **Safe targets**: optimizer config, scheduler addition

3. **/repo/bandgap/train/eval_final.py** — Reproduction evaluation:
   - Configured: SEED=1024, MAX_EPOCHS=300, PATIENCE=100, LR=1e-4, BATCH_SIZE=128
   - Imports model/loss from model_mine.py
   - **Safe targets**: LR, epochs, patience, seed, optimizer config, scheduler

4. **/repo/bandgap/dataset/dataset.py** — Dataset loading:
   - Uses pymatgen Composition + matminer ElementProperty (magpie preset)
   - Featurizes into 73-dim composition vector + 132-dim total features
   - State: 0=HSE, 1=GGA
   - **DO NOT MODIFY** — test data/splits must remain unchanged

## Metric Parser
- Metrics extracted from stdout: MAE, RMSE, tau_b
- Also saved in /repo/bandgap/reproduction_result.json
- Compat metric: MAE (lower is better)
- Guardrails: RMSE (lower, 5 pct), tau_b (higher, monitor)

## Baseline
- MAE=0.566, RMSE=0.742, tau_b=0.637
- Seed=1024, 300 epochs, patience=100, lr=1e-4
- Training time: ~20-40 min on A100

## Repository Layout
- /repo/bandgap/data/ — Dataset CSVs (bandgap.csv, hse.csv, pbe.csv, etc.)
- /repo/bandgap/pt/ — Saved model checkpoints
- /repo/bandgap/run.sh — Shell script
- /repo/mol_qm/ — Separate molecular QM task (not the evaluation target)

## Known Levers (from manifest)
- Extended training epochs (200->300), patience (50->100), seed selection
- alpha_abs=1.0 and alpha_rank=5e-3 tunable (paper Table 3)
- KL weight (1e-3), dropout rates, activation functions

## Risky Files (DO NOT MODIFY)
- /repo/bandgap/dataset/dataset.py — Changes split or featurization
- /repo/bandgap/data/*.csv — Changes labels or test data
- Any evaluation metric computation in eval_final.py
