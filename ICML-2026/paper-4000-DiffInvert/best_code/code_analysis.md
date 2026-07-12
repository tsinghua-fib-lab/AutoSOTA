# Code Analysis for Paper 4000 (TIED) Optimization

## Evaluation Pipeline
1. `test_mnist_classification.py --config configs/affnist/logsumexp_diffusion.yaml`
   - Loads dataset (affNIST), classifier (ResNet18 MNIST), energy (ImageClassifierEnergy), sampler (EnergyDiffusionSampler)
   - Wraps in LitModel with task="test/image_classification"
   - Runs trainer.test() -> computes test accuracy
   - Saves canonicalized images to experiments/repro_mnist_imgs/affNIST/
2. `fid_score.py --true <padded_mnist.npy> --fake <canonicalized.npy> --model lenet -c 0`
   - Computes FID between padded MNIST (training distribution) and canonicalized test images

## Key Files
- `configs/affnist/logsumexp_diffusion.yaml` - Diffusion sampler config (steps=50, num_mc=2, num_hp=64, T=1.0)
- `src/group_sampler.py` - EnergyDiffusionSampler (diffusion process, score estimation, hard argmin selection)
- `src/lit_model.py` - LitModel (wraps sampler+classifier, has ensemble code path but disabled)
- `src/energies.py` - ImageClassifierEnergy (E = -logsumexp(logits))
- `src/pretrained_models.py` - ResNet18Classifier wrapper
- `test_mnist_classification.py` - Main eval script
- `fid_score.py` - FID computation with LeNet5 features

## Critical Bottlenecks
1. Hard argmin hypothesis selection (group_sampler.py line ~345): only 1/64 hypotheses used
2. Ensemble code path exists in lit_model.py (lines 67-87) but `ensemble=False` by default
3. Uniform averaging in ensemble (no softmin weighting)
4. FID saving in ensemble path saves ALL hypotheses images (64x), polluting FID

## Baseline Metrics (Iteration 0)
- Accuracy: 82.53% (paper: 82.64+-0.11)
- FID: 5.04 (paper: 5.05)
- Runtime: 37.25 min (paper: 37 min on A6000)

## Safe Modification Targets
1. `src/group_sampler.py`: Add return_all parameter, return all hypotheses + energies
2. `src/lit_model.py`: Add softmin-weighted ensemble averaging, fix FID save
3. `test_mnist_classification.py`: Pass ensemble config to sampler and LitModel
4. `configs/affnist/logsumexp_diffusion.yaml`: Add ensemble parameters

## Red-Line Safe Files (DO NOT MODIFY)
- `fid_score.py` - Metric computation
- `src/energies.py` - Energy definition (unless adding features)
- `src/pretrained_models.py` - Classifier (unless adding feature hooks)
- `src/datasets.py` - Test data loading
- Datasets under /datasets/ - Test data

## Repo State
- Git repo at /repo, branch main
- Baseline commit: 18200a4
- _baseline tag: NOT set (git tag), but baseline is commit 18200a4
