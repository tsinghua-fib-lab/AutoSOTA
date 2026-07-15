# Code Analysis — Paper 5631: DLL (Diffusion Last Layer)

## Repository
- URL: https://github.com/sungwpark/dll-no
- Commit: af3837c302aef46fdf5745be3862cc8c07abfce3

## Pipeline Overview
Two-stage training:
1. OperatorEncoder (OE): FNO1d backbone + output embedder
   - Maps x -> features, y -> weights w
   - Reconstructs y_hat = features @ w
   - 100 epochs, batch 64, lr 1e-3, CosineAnnealingLR to 0

2. DLL: CondMLP flow-matching on weight space
   - Condition embedder: FnoEmbedding1d (x -> conditioning features)
   - Flow matching: v_pred = model(x_t, t, cond), v_target = x1 - x0
   - Inference: sample weight w via ODE, reconstruct y = features @ w
   - 100 epochs, batch 64, lr 1e-3, CosineAnnealingLR to 0

## Key Observations
1. CosineAnnealingLR with eta_min=0.0: final ~20% epochs at near-zero LR
2. Zero weight_decay: potential overfitting of CondMLP (3x512 hidden)
3. Uniform t ~ U(0,1) sampling: no emphasis on critical denoising regime
4. No time-dependent loss weighting
5. NFE=10 is low; Heun solver available but unused in eval
6. Stochastic evaluation has inherent variance from random noise in sampling
7. test_samples_per_example=32 gives moderate MC variance

## Fast Eval Command (uses pretrained checkpoints)
operatorencoder.pretrained_ckpt=/repo/checkpoints/SBurgers_AE/oe_best.ckpt dll.pretrained_ckpt=/repo/checkpoints/SBurgers_GEN/dll_best.ckpt
