# Code Analysis for Paper 19: F-DPMSolver SOTA Optimization

## Evaluation Path
- evaluate.sh: Generates 50K CIFAR-10 samples via torchrun then computes FID via cleanfid
- FID computation uses clean-fid library with TF InceptionV3 stats for CIFAR-10 train split
- Key eval parameters: NFE=4, order=1, algorithm=F-DPMSolver, model=CIFAR10-uncond, batch=64

## Core Inference Path
- main.py: Entry point, arg parsing, distributed sampling loop
- sampler.py: All sampler algorithms + schedule + ODESampler_onestep
- utils/load_model.py: EDM model loading with eps-prediction wrapper

## Safe Modification Targets
1. sampler.py: ODESampler_onestep(), get_schedule(), Forward_DPMSolver()
2. main.py: Add new CLI args for rho, sigma_min, sigma_max, gamma
3. evaluate.sh: Can change sampler parameters (NFE, order, algorithm)

## Red Lines
- Do NOT modify model weights or model loading
- Do NOT modify FID computation (clean-fid)
- Do NOT change test data (CIFAR-10 train split for stats)
- Do NOT hard-code metric values
