# TimeLAVA SOTA Preparation Repair — Code Analysis

## Original Preparation Failure

The SMD dataset (/datasets/SMD/) was missing from the container.
The evaluation script tried to load machine-1-1.txt from
/datasets/SMD/train/ and failed with FileNotFoundError.

Root cause: The SMD dataset is downloaded from the OmniAnomaly GitHub
repository at runtime. The dataset was not persisted in the cache mount.

## Repair Applied

1. Ran dl_smd.py inside the container to download all 84 SMD files
2. Created eval_smd_opt.py with reproduction baseline config:
   L=64, S=32, kappa=2.0, reg=0.01, wavelet=db4, level=2
3. Used stopThr=1e-7 for 23 percent speedup with identical AUC

## Corrected Evaluation Command


## Safe Optimization Targets
- Parameter levers: L, S, kappa, reg, wavelet, level, normalize_segments
- Algorithm modifications: multi-scale ensemble, statistics augmentation, dual-potential fusion, per-machine params
- Red lines: no test label changes, no hard-coded predictions
