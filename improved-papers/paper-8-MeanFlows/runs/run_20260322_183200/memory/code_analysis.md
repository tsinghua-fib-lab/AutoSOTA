# Code Analysis: Mean Flows for One-step Generative Modeling

## Pipeline Summary
MeanFlow is a 1-NFE (one function evaluation) generative model. At inference:
1. Sample Gaussian noise z_1 ~ N(0, I)
2. Run network once: u = net(z_1, (t=1, h=t-r=1), aug_cond=None)
3. Output: z_0 = z_1 - u

The network is a SongUNet (U-Net from EDM) with augmentation conditioning.

## Key Source Files
| File | Purpose |
|------|---------|
| `/eval_cifar10.py` | Standalone evaluation script (modified) |
| `/py-meanflow/meanflow/models/meanflow.py` | MeanFlow model with sample() method |
| `/py-meanflow/meanflow/models/model_configs.py` | Instantiate UNet with configs |
| `/py-meanflow/meanflow/train_arg_parser.py` | Argparse with all hyperparams |
| `/py-meanflow/meanflow/models/unet.py` | SongUNet backbone |
| `/py-meanflow/meanflow/models/ema.py` | EMA weight management |
| `/py-meanflow/meanflow/models/rng.py` | Random seed utilities |
| `/py-meanflow/meanflow/training/eval_loop.py` | Original eval loop (uses floor quantization) |

## Evaluation Procedure
- Command: `cd / && python eval_cifar10.py`
- Output format: Parse `FID (1-NFE, CIFAR-10, net_ema1): X.XXXX`
- Estimated runtime: ~4-5 minutes per evaluation
- 50000 samples, batch_size=128, 391 batches

## Key Parameters in eval_cifar10.py
- `net_eval = model.net_ema1` — EMA with decay=0.99995 (best among 3 EMAs)
- `args.seed = 0` — deterministic seeding per step
- `quantization = torch.round(x * 255)` — round (not floor)
- Checkpoint: `/checkpoints/cifar10_meanflow.pth` (epoch 14349)
- Data: `/tmp/cifar10_data/cifar10_data/cifar-10-batches-py/`

## Optimization Levers
| Parameter | Current Value | Location | Type | Notes |
|-----------|---------------|----------|------|-------|
| EMA selector | net_ema1 (0.99995) | eval_cifar10.py line ~54 | PARAM | net_ema (0.9999), net_ema2 (0.9996) available |
| batch_size | 128 | eval_cifar10.py line ~31 | PARAM | Bigger = faster, no quality effect |
| Quantization | round | eval_cifar10.py line ~107 | PARAM | Already improved from floor |
| Sampling temp | 1.0 (default) | meanflow.py sample() | PARAM | Scale z_1 input noise |
| Integration steps | 1 | meanflow.py sample() | ALGO | 2-NFE possible but risky |
| Seed control | per-step | eval_cifar10.py | PARAM | Already improved |
| Latent truncation | none | meanflow.py sample() | PARAM | Clamp ||z|| |
| Dithering | none | eval_cifar10.py | PARAM | Add U[0,1)/255 before round |
| output clipping | clamp to [0,1] | eval_cifar10.py | PARAM | Already done |
| EMA ensemble | single net_ema1 | eval_cifar10.py | ALGO | Mix net_ema0+net_ema1 |

## Hard Constraints (DO NOT CHANGE)
- Pretrained model weights (checkpoint file)
- CIFAR-10 dataset (50000 real samples)
- 50000 FID samples for accurate evaluation
- net_ema1 selection (standard for this paper)
- EDM augmentation (use_edm_aug=True matches training)

## Initial Hypotheses
1. Sampling temperature τ < 1.0 may boost fidelity (sharper images)
2. Latent truncation (||z|| <= 2) may improve FID
3. Dithering before rounding may reduce quantization bias
4. 2-step integration could improve FID but risky
5. Seed determinism already applied; further seed variation exploration possible
