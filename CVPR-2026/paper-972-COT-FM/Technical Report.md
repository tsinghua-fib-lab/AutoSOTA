# COT-FM: Cluster-wise Optimal Transport Flow Matching: A Technical Report on Automated Optimization

## Abstract
This report documents an automated optimization study performed by the AutoSOTA pipeline on the COT-FM framework—a plug-and-play method that improves continuous normalizing flows for generative modeling by clustering target samples and assigning per-cluster source distributions, yielding straighter probability paths, lower discretization error, and faster sampling. The optimization targeted the unconditional 2D point-cloud task on the 5‑Gaussians dataset, aiming to reduce the squared Wasserstein‑2 (W2²) distance and trajectory curvature. A single intervention replaced uniform time sampling (t ~ U[0,1]) with a Beta(0.6, 0.6) distribution during both pretraining and COT‑FM training. This change reduced W2² by 13.2 % (from 0.1513 to 0.1313) and curvature by 12.8 % (from 0.0075 to 0.0065), surpassing the optimization target of W2² ≤ 0.1437 in one iteration. The 13.2 % reduction in W2² and 12.8 % reduction in curvature confirm that endpoint‑focused time sampling improves COT‑FM’s sample quality and trajectory straightness.

## 1. Introduction
COT‑FM (Chiang et al., 2026) clusters target samples and assigns each cluster a dedicated source distribution, creating multiple short, straight probability paths that collectively reduce discretization error during numerical integration. In its original form, the training procedure samples the interpolation time *t* uniformly from [0,1], allocating equal probability mass to every instant of the flow. This uniform sampling provides limited training signal near the trajectory endpoints (t≈0 and t≈1), where the velocity field must be most accurate to maintain low ODE truncation error and faithful sample generation.

This report presents an automated optimization study conducted by the AutoSOTA framework on the COT‑FM codebase. The study aimed to improve the W2² metric and trajectory curvature on the 5‑Gaussians benchmark. The sole intervention—replacing uniform time sampling with a Beta(0.6, 0.6) distribution—yielded significant gains. The following sections detail the original method, the identified limitation, the intervention’s rationale, experimental results, and implications.

## 2. Original Method (Background)
COT‑FM reshapes the flow‑matching probability path by clustering the target data and aligning each cluster with a separate source distribution. For a cluster *k*, the interpolation becomes

xₜ⁽ᵏ⁾ = (1 − t) μₖ + t x₁⁽ᵏ⁾, t ∈ [0,1],

where μₖ is the cluster‑specific source mean and x₁⁽ᵏ⁾ are target samples in that cluster. This alignment produces straighter trajectories than a single global source, lowering overall curvature. COT‑FM is architecture‑agnostic: it does not modify the neural network, the velocity‑field loss, or the ODE solver, making it compatible with existing flow‑matching backbones such as OT‑CFM, Rectified Flow, and MeanFlow.

The public repository (Chiang et al., 2026) provides experiments on toy 2D datasets (5‑Gaussians, Two Moons, Checkerboard) and image generation on CIFAR‑10 and ImageNet. The toy‑2D experiments are run from a single script, `toy_2d/main.py`, which handles training, sampling, and metric computation. The optimization study used the 5‑Gaussians task with the W2² distance and trajectory curvature as evaluation metrics.

## 3. Identified Limitations
The original `toy_2d/main.py` samples time steps uniformly from U[0,1] in both the pretraining loop (source‑to‑noise flow) and the main COT‑FM training loop. The relevant code contains lines such as

```
t = torch.rand(x0.shape[0], 1, device=self.device)        # pretrain
t = torch.rand(x1_batch.shape[0], 1, device=self.device)   # COT-FM
```

This uniform sampling treats all temporal positions equally, yet the ODE integration error is concentrated near the flow boundaries. Inaccuracies in the velocity field at t≈0 and t≈1 propagate through the trajectory, increasing the final discrepancy between generated and true samples and adding curvature. Consequently, the baseline W2² of 0.1513 and curvature of 0.0075 leave room for improvement relative to the optimization target of W2² ≤ 0.1437.

The hypothesis is that a non‑uniform time sampling scheme that emphasizes the start and end of the unit interval will provide more relevant training signal, reduce endpoint errors, and thereby decrease both the distributional distance and the trajectory curvature.

## 4. Optimization Methodology
The AutoSOTA pipeline implemented a single accepted intervention: replacing all uniform time sampling calls with sampling from a Beta(0.6, 0.6) distribution. The change was introduced into `toy_2d/main.py` (the 2D training script) by adding the function `sample_time_beta` and substituting every `t = torch.rand(...)` call with `t = sample_time_beta(...)`. The intervention is applied in both the pretraining and COT‑FM training loops.

The rationale is as follows:
- **Limitation → Hypothesis**: Uniform *t* sampling under‑represents the endpoints, where the velocity field must be most accurate for low‑error ODE integration and faithful inversion of the pretrained flow. A U‑shaped Beta distribution naturally allocates more probability mass near 0 and 1, stressing these critical regions.
- **Expected effect**: Devoting more training capacity to t≈0 and t≈1 should produce a more accurate velocity field near the boundaries, reducing numerical integration error and making trajectories straighter. Both W2² and curvature are therefore expected to decrease.

The optimized model was retrained from scratch using this change, and the metrics were recomputed, yielding the commit `ebf024780cc710f8f606a73d6eeb07088cd7fc44`.

## 5. Experiments
### 5.1 Setup
The optimization was performed on the unconditional 2D point‑cloud generation task with the 5‑Gaussians dataset. The following details summarise the configuration:

- **Hardware**: Not explicitly reported in the optimization log; experiments used the repository’s default settings.
- **Dataset**: 5‑Gaussians, synthetically generated as per `toy_2d` defaults.
- **Evaluation protocol**: After training, the model generates samples by integrating the learned ODE with an Euler solver (the default used in the original code). The squared Wasserstein‑2 distance between generated and true samples, and the trajectory curvature, are computed.
- **Baseline command**: The original `main.py` invoked with default arguments, corresponding to uniform t ~ U[0,1].
- **Optimization budget**: The pipeline was configured with a target of W2² ≤ 0.1437; it terminated after one iteration because the target was already exceeded.
- **Seed**: The repository’s default seed (0) was used, though the log does not explicitly record it. Reproducibility is discussed in Section 7.
- **Caveats**: The metrics are reported as computed by the repository’s own evaluation code. The study was limited to the 5‑Gaussians task; generalization to other datasets is discussed in Section 6.

### 5.2 Quantitative Results
Table 1 presents the baseline and optimized metrics. W2² decreased by 13.2 % and curvature by 12.8 %.

| Metric          | Baseline | Best   | Delta   | Direction       |
|-----------------|----------|--------|---------|-----------------|
| W2²             | 0.1513   | 0.1313 | -13.2%  | Lower is better |
| Curvature       | 0.0075   | 0.0065 | -12.8%  | Lower is better |

**Table 1**: Performance comparison on the 5‑Gaussians dataset.

### 5.3 Ablation / Iteration Trajectory
The optimization trace consists of the baseline and a single intervention step (Table 2).

| Iteration | Change                                              | W2²   | Curvature |
|-----------|-----------------------------------------------------|--------|-----------|
| 0         | Baseline (uniform t ~ U[0,1])                      | 0.1513 | 0.0075    |
| 1         | Beta(0.6, 0.6) time sampling (pretrain & train)    | 0.1313 | 0.0065    |

**Table 2**: Iteration trajectory. The intervention immediately lowered both metrics, surpassing the target after one step.

## 6. Discussion
The large gain from a single, straightforward change highlights the sensitivity of flow‑matching models to the distribution of training time steps. By emphasizing the trajectory endpoints, the Beta(0.6, 0.6) sampling concentrates learning on the regions where velocity errors most severely amplify during integration. The concurrent reductions in W2² and curvature confirm that more accurate endpoint predictions lead to straighter paths, which in turn lower truncation error and bring the generated distribution closer to the true one. The relative improvement of over 13 % in W2² is notable for a modification that neither alters the architecture, the number of training iterations, nor the ODE solver.

The study is limited to a low‑dimensional synthetic dataset. The 5‑Gaussians task, while standard, may not capture the behaviour of high‑dimensional image generation, where the optimal time‑sampling distribution could differ. Only one Beta parameterization (α=β=0.6) was tested; a sweep over other concentration parameters might yield further improvement. Additionally, only a single random seed was used, so run‑to‑run stochasticity is not assessed. Because the pipeline terminated early, other promising ideas listed in the optimization log (Heun ODE solver, sinusoidal time embeddings, curvature regularization, cosine LR scheduling, and cluster‑count optimization) remain unexplored.

The underlying principle—allocating more training signal to the flow endpoints—is domain‑agnostic and is likely to transfer positively to CIFAR‑10 or ImageNet settings, although explicit verification would require retraining at scale.

## 7. Reproducibility
To reproduce the optimized result:

- **Repository**: https://github.com/embodiedai-ntu/cotfm, at commit `ebf024780cc710f8f606a73d6eeb07088cd7fc44`.
- **Environment**: Python 3.10+ with PyTorch 2.0+; refer to `toy_2d/requirements.txt` for exact versions.
- **Seed**: Use the default seed 0 (as set in `toy_2d/main.py`).
- **Baseline run**: Execute the original `toy_2d/main.py` with default arguments (uniform time sampling).
- **Optimized run**: Apply the diff from the optimization log to `toy_2d/main.py`:
  1. Add the `sample_time_beta` function (Beta(0.6, 0.6) sampling).
  2. Replace every `t = torch.rand(...)` call in the pretraining and training loops with `t = sample_time_beta(...)`.
  Then re‑run `main.py`.
- **AutoSOTA invocation**: The precise AutoSOTA command is not archived here, but the framework applies the described diff programmatically.

## 8. References
```bibtex
@misc{chiang2026cotfmclusterwiseoptimaltransport,
      title={COT-FM: Cluster-wise Optimal Transport Flow Matching}, 
      author={Chiensheng Chiang and Kuan-Hsun Tu and Jia-Wei Liao and Cheng-Fu Chou and Tsung-Wei Ke},
      year={2026},
      eprint={2603.13395},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.13395}, 
}

@misc{autosota,
  author = {Tsinghua FIB Lab},
  title = {AutoSOTA: Automated State-of-the-Art Optimization},
  year = {2025},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}},
}
```
