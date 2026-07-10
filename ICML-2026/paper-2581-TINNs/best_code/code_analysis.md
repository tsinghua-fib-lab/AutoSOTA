# TINN Allen-Cahn Code Analysis (Paper 2581)

## Evaluation Path
- Command: `cd Allen-Cahn && python3 eval.py --seed 0`
- `eval.py` is a standalone script that trains the TINN model and evaluates Relative L2 Error against reference data `new_AC.mat`
- Metric parsing: Parse stdout for `Metric: Relative L2 Error = <value>` line

## Training/Inference Path
- `eval.py` contains the full training loop (LM optimizer, 30K iterations)
- Model: TINN class (time encoder + periodic embedding spatial network)
- Time encoder: [1,10,10,5] (input t, 2 hidden layers, 5-dim output for spatial modulation)
- Spatial network: [2,20,20,1] (periodic embedding input, 2 hidden layers, scalar output)
- Periodic embedding: [cos(πx), sin(πx)] — only first harmonic
- LM optimizer parameters: mu=10, mu_update=2, div_factor=1.3, mul_factor=1.7

## Config Path
- `eval.py` uses argparse with defaults matching paper settings
- Key configurable params: --seed, --epochs (30000), --Nc (10000), --Nic (500), --lambda-ic (20.0), --mu-init (10.0), --div-factor (1.3), --mul-factor (1.7), --mu-min (1e-12), --mu-max (1e8)

## Metric Parser
- Final output line: `Metric: Relative L2 Error = <value>`
- Baselines: Relative L2 Error = 3.6e-06, Training Time ~0.16h

## Reusable Resources
- `/repo/Allen-Cahn/new_AC.mat` (747KB) — Reference PDE solution
- No external datasets or model weights needed

## Risky Files
- `Allen-Cahn/TINN-AC.py` — Original training script (used for baseline)
- `Allen-Cahn/eval.py` — Current evaluation script (target for optimization)
- `Allen-Cahn/new_AC.mat` — DO NOT MODIFY (reference data)

## Safe Modification Targets
1. `Allen-Cahn/eval.py` lines 130-140 (IC lambda, epochs, N_coll, N_ic, LM params)
2. Training loop in eval.py: LM update logic, checkpointing, resampling
3. Model architecture: periodic embedding, time encoder dimensions
4. Loss function: IC weighting schedule
5. LM optimizer: mu update schedule, line search addition

## Key Observations
1. No best-model checkpointing — uses final params after all training
2. Fixed collocation points with only reactive resampling (val_tot/loss > 5)
3. LM update every 2 steps (too frequent, may oscillate)
4. No line search — dp applied directly without checking loss reduction
5. Only first harmonic in periodic embedding (could benefit from higher harmonics)
6. JAX float64 required for numerical stability of JVP-based Jacobian
7. 221 total parameters (very small model)
