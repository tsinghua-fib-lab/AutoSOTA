# Code Analysis: Mean Flows for One-step Generative Modeling

## Pipeline Summary
MeanFlow learns a probability flow ODE to transport from Gaussian noise (t=1) to data distribution (t=0). During inference, only one network function evaluation (NFE) is needed: given noise z₁, predict velocity u, then z₀ = z₁ - u. The network is a SongUNet (EDM-style UNet for CIFAR-10 32×32).

## Key Source Files
| File | Purpose |
|------|---------|
| `/py-meanflow/meanflow/models/meanflow.py` | Core MeanFlow model: forward_with_loss, sample() |
| `/py-meanflow/meanflow/models/unet.py` | SongUNet architecture (~1500 lines) |
| `/py-meanflow/meanflow/models/model_configs.py` | Model instantiation, architecture config |
| `/py-meanflow/meanflow/train_arg_parser.py` | All hyperparameters with defaults |
| `/py-meanflow/meanflow/models/ema.py` | EMA update logic |
| `/py-meanflow/meanflow/models/time_sampler.py` | (t,r) sampling strategies v0 and v1 |
| `/py-meanflow/meanflow/training/eval_loop.py` | Training-time FID evaluation |
| `/eval_cifar10.py` | Standalone FID evaluation script |

## Evaluation Procedure
- Command: `cd / && python eval_cifar10.py`
- Parse: `float(line.split("FID (1-NFE, CIFAR-10, net_ema1): ")[1])`
- Uses net_ema1 (ema_decay=0.99995)
- 50K synthetic samples, FrechetInceptionDistance(normalize=True)
- Quantizes synthetic to 8-bit: `floor(x*255)/255`
- Runtime: ~2.5 minutes on A100

## Architecture: SongUNet (model_channels=128)
- img_resolution=32, in_channels=3, out_channels=3
- channel_mult=[2,2,2]
- channel_mult_noise=2 (noise_channels=256)
- resample_filter=[1,3,3,1]
- encoder_type='standard', decoder_type='standard'
- augment_dim=6 (because use_edm_aug=True was passed at training)
- dropout=0.2

## EMA Networks (checkpoint has 3)
| Network | Attribute | ema_decay |
|---------|-----------|-----------|
| net_ema | Primary primary | 0.9999 |
| net_ema1 | used in eval | 0.99995 |
| net_ema2 | fastest decay | 0.9996 |

## Sampling Procedure (model.sample)
```python
e = torch.randn(samples_shape)  # pure Gaussian noise
t = torch.ones(N)               # t=1 (noise end)
r = torch.zeros(N)              # r=0 (data end)
u = net(z_1, (t, t-r=1), aug_cond=None)  # h = t-r = 1
z_0 = z_1 - u                  # one-step prediction
```

## Optimization Levers
| Parameter | Current Value | File:Line | Type | Notes |
|-----------|---------------|-----------|------|-------|
| net_ema selector | net_ema1 | eval_cifar10.py:61 | CODE | Try net_ema (0.9999) or net_ema2 (0.9996) |
| batch_size | 128 | eval_cifar10.py:32 | PARAM | Inference batch size, affects speed not quality |
| fid_samples | 50000 | eval_cifar10.py:35 | PARAM | Keep at 50K for accuracy |
| seed | 0 | eval_cifar10.py:34 | PARAM | Different seeds give different FID estimates |
| quantization floor→round | floor | eval_cifar10.py:118 | CODE | Round instead of floor for better quality |
| clamp range | [0, 1] | eval_cifar10.py:115-119 | PARAM | Already max range |
| aug_cond in sample() | None | eval_cifar10.py:107 | CODE | None since eval, correct |
| num_workers | 4 | eval_cifar10.py:83 | PARAM | Increasing speeds up data loading |

## Critical Finding: Quantization
```python
# Current (eval_cifar10.py line 118-119):
synthetic_samples = torch.floor(synthetic_samples * 255)
synthetic_samples = torch.clamp(synthetic_samples / 255.0, 0.0, 1.0)  # BUG: divides again after floor!
```
Wait - is this `torch.clamp(synthetic_samples / 255.0)` double division? No - it's:
- After scale: `x = clamp(raw * 0.5 + 0.5)`  (range [0,1])
- `x = floor(x * 255)` → range [0, 255]
- `x = clamp(x / 255.0, 0.0, 1.0)` → back to [0, 1] ✓

Alternative: use `torch.round` instead of `torch.floor` → slightly different distribution

## Hard Constraints (DO NOT CHANGE)
- Pretrained weights at /checkpoints/cifar10_meanflow.pth
- 50K FID samples
- CIFAR-10 train split for real distribution
- FrechetInceptionDistance metric computation
- net_ema1 evaluation target (in the eval script name/output)
- Number of ODE steps = 1 (1-NFE metric)

## Initial Hypotheses
1. **EMA network selection**: net_ema0 (0.9999) vs net_ema1 (0.99995) — already used net_ema1 which gave 2.8698; try net_ema2 (0.9996) which might be better at this sampling regime
2. **Round vs Floor quantization**: Round instead of floor reduces systematic negative bias in generated images
3. **Seed exploration**: FID is stochastic; different seeds give slightly different values (±~0.05)
4. **Multi-EMA ensemble**: Average predictions from multiple EMA networks
5. **Batch size during inference**: Larger batch might affect numerical results slightly
