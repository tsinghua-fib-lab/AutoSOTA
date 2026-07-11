# CycMit-MRI Code Analysis for SOTA Optimization

## Evaluation Path
- **Entry point**: `eval.py` -> `main()`
- **Flow**: Load config -> Load data/mask/model -> Noise jittering -> PGD attack -> Cyclic Mitigation -> Metrics
- **Metric parser**: Parse stdout for `METRICS:PSNR=<value>,SSIM=<value>`
- **Primary metric**: PSNR (higher is better), Guardrail: SSIM (>=0.91)

## Key Files
| File | Role | Safe to modify |
|------|------|----------------|
| `eval.py` | Evaluation entry point | Yes - metric computation, config loading |
| `Config.yaml` | All hyperparameters | Yes - parameter values |
| `src/Utils.py` | Core mitigation logic, loss functions, metrics | Yes - algorithmic changes |
| `src/Unrolled_Network.py` | MoDL unrolled network | No - model architecture |
| `src/ResNet.py` | ResNet backbone | No - pretrained weights |
| `src/DF.py` | Data fidelity (CG) | No |
| `DataLoader.py` | Data loading | No |
| `Mitigation.py` | Standalone mitigation script | N/A (not used by eval) |

## Critical Function: Cyclic_Mitigation (src/Utils.py:194-248)
- **Input**: zf_p (attacked zero-filled), label, coil, masks
- **Loop**: 100 iterations of:
  1. ABA_detect: Model forward with cyclic mask stages, returns reconstructions
  2. Loss computation: L2_Loss(zf_temp_ksp, err_ksp[...,-1]) -- uses ONLY last cyclic stage
  3. Backward + reverse PGD projection
  4. PSNR/SSIM computed on Recons_mit[:,:,0] -- uses ONLY first cyclic stage
- **BUG**: PSNR uses first cyclic stage ([:,:,0]) but loss uses last stage ([...,-1]). The last stage should give better reconstruction.

## Known Levers
- `alpha_mitigate`: Step size (0.005) -- primary tuning parameter
- `epsilon_proj`: Projection radius (0.01) -- controls recovery capacity
- `iterations_mitigate`: 100 -- more = better but slower
- `alpha_scheduler`: Linear or Exp -- convergence trajectory
- `cyclic_stages`: 2 -- number of cyclic mask stages
- `noise_jittering_std`: 6 -- noise level for jittering
- `mu`: 0.1813 -- data fidelity weight in MoDL

## Risky Files
- `./data/`: Test data -- DO NOT MODIFY
- `./BestModel/checkpoint.pth`: Pretrained model -- DO NOT MODIFY
- `getpsnr()`, `getssim()` in Utils.py: Metric functions -- DO NOT MODIFY

## Container Setup
- GPU devices: 6,7
- PyTorch 2.1.0+cu121
- Config.yaml device: cuda:0

## Baseline
- iter=0: PSNR=34.306, SSIM=0.919
- Paper target: PSNR=35.14, SSIM=0.92
