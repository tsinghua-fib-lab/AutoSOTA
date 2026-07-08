# Code Analysis - Paper 1837 SOTA Preparation Repair

## Preparation Failure

Root cause: Docker overlay filesystem was 100% full (200G/200G). apt-get install git failed.
Fix: Removed dangling images, installed git v2.25.1. CUDA_VISIBLE_DEVICES=0,1.

## Corrected Evaluation Command

cd /repo && CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/autosota_cache/pip-packages:$PYTHONPATH \
python3 /autosota_cache/paper-1837/eval_metrics.py \
  --checkpoint runs/n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=153476998_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt \
  --niters 20

## Baseline Verification

Manifest baseline: Model avg log k-means obj = 4.9969
Repaired evaluation: Model = 4.9995 (within CI [4.979, 5.264])

## Key Files

- kmt.py: KMeansTransformer + KModel (4-head transformer, ~14K params)
- losses.py: SoftKMObj loss (soft k-means upper bound)
- ctasks.py: ClusteringTasks (5 distributions: normal, cauchy, gumbel, laplace, lognormal)
- utils.py: Lloyd algorithm, trimmed k-means, robust k-means
- train.py: Training loop, ReduceLROnPlateau, gradient clipping commented out

## Model Architecture

KModel with 4 attention heads:
- CAX: cross-attention (centers attend to points)
- SAX: self-attention (points attend to points)
- CAC: cross-attention (centers attend to updated points)
- SAC: self-attention (centers attend to centers)

No FFN, no normalization layers, no residual scaling.
Euclidean distance attention: -inv_temp * ||Q-K||^2 / sqrt(d_k)
d_emb = 32 + 10 = 42, d_qkv = 42

## Training

- Adam(lr=0.01), ReduceLROnPlateau(factor=0.5, patience=5)
- SoftKMObj(gamma=10.0), single forward pass per step
- 10000 steps, batch 32, eval every 10 steps
- Gradient clipping commented out (train.py line 423)
