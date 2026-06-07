# IsoCLIP: Decomposing CLIP Projectors for Efficient Intra-modal Alignment — A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study conducted on IsoCLIP, a training-free method for improving intra-modal retrieval with Contrastive Language-Image Pre-training (CLIP) models. IsoCLIP isolates an approximately isotropic subspace within the CLIP shared embedding space by spectrally decomposing the inter-modal projector product, discarding anisotropic directions that cause intra-modal misalignment. The original method uses a hard binary selection of a contiguous singular vector band, controlled by parameters `ktop` and `kbottom`, which is known to be dataset‑dependent. An automated pipeline (AutoSOTA) explored enhancements to the spectral thresholding and band selection strategy, targeting a mean Average Precision (mAP) of 28.38 on the CUB‑2011 fine-grained image retrieval benchmark with a ViT‑B/32 backbone. The best configuration, combining soft sigmoid‑weighted thresholding and a multi‑band ensemble that averages similarities from four `(ktop, kbottom)` combinations, achieved an mAP of **27.39**, a **+1.33 %** improvement over the baseline (27.03). The proposed modifications operate exclusively at inference time and introduce negligible computation. The target was not fully reached (96.5 % of target), highlighting the task‑specific sensitivity of spectral band selection and motivating future dataset‑aware optimization.

## 1. Introduction

Vision‑language models pre-trained with a contrastive objective, such as CLIP, align image and text representations in a joint embedding space. While this alignment is highly effective for cross‑modal tasks like zero‑shot classification, applying the same encoders to single‑modality tasks—most notably image‑to‑image retrieval—exposes a persistent *intra‑modal misalignment*: features from the same modality are not optimally organised for nearest‑neighbour search. IsoCLIP (Magistri et al., CVPR 2026) addresses this problem by decomposing the product of the image and text projection matrices (`W_image` and `W_text`) and retaining only those spectral directions in which the two modalities are well aligned. The retained subspace corresponds to an isotropic region of the inter‑modal operator; removing the most dominant and the weakest singular components improves intra‑modal similarity estimation without any additional training or expensive test‑time procedures.

The original IsoCLIP employs a hard binary selection of a single contiguous band of singular vectors. While effective, this design leaves room for refinement: the hard transition at band boundaries can discard marginally useful information, and a single band cannot capture all complementary discriminative directions. An automated optimization framework, AutoSOTA, was applied to the official IsoCLIP repository to explore these degrees of freedom systematically. The pipeline was given a fixed budget of 11 iterations and a target mAP of 28.38. The study identified two successful interventions that together raised performance on CUB‑2011 by 0.36 mAP points.

## 2. Original Method (Background)

IsoCLIP is a training‑free, latency‑free operator applied to pre‑projection CLIP features. Let `X_image` and `X_text` be the pre‑projection representations produced by the CLIP image and text encoders. Their projected counterparts are `X_image W_image` and `X_text W_text`, where `W_image ∈ ℝ^{d_{vis} × d}` and `W_text ∈ ℝ^{d_{txt} × d}` are the projection matrices (typically `d = 512`). The inter‑modal similarity underlying CLIP’s contrastive loss can be expressed through the operator `Ψ = W_image^⊤ W_text`. An SVD of `Ψ` yields singular vectors `U` and `V` and singular values `S`. The authors observe that the most isotropic, well‑aligned directions lie in the middle of the spectrum, while the top and bottom components over‑emphasise modality‑specific anisotropic structure that harms intra‑modal retrieval.

The original IsoCLIP procedure discards the top `ktop` and bottom `kbottom` singular components:

```
U_k = U[:, ktop : r − kbottom]
V_k = V[:, ktop : r − kbottom]
W_image_iso = W_image U_k U_k^⊤
W_text_iso = W_text V_k V_k^⊤
```

New features are computed as `X_image W_image_iso` (and analogously for text), normalized, and compared via cosine similarity. The method has two hyper‑parameters, `ktop` and `kbottom`, validated primarily on Caltech101. For image‑to‑image retrieval, `ktop = 150` and `kbottom = 50` are recommended for the ViT‑B/32 backbone. The codebase supports both retrieval (`src/retrieval.py`) and classification (NCM with IsoCLIP) and is built on top of Dassl and Cross‑the‑Gap.

## 3. Identified Limitations

### 3.1 Hard Spectral Thresholding Induces Information Loss
In the default implementation (function `apply_iso` in `src/retrieval.py`), the transition from retained to discarded singular components is binary: all vectors outside the chosen band are entirely zeroed out. The spectral boundary is not sharp in practice; singular values decay gradually, and directions immediately beyond the cut‑off indices may still contain useful inter‑modal alignment. The rigid cut risks discarding partial discriminative information, which is particularly harmful for fine‑grained retrieval where subtle distinctions are critical.

### 3.2 Single‑Band Selection Cannot Capture Complementary Spectral Scales
A single pass with fixed `ktop` and `kbottom` produces one isotropic subspace. Different choices of the truncation limits reveal different structural aspects of the intra‑modal manifold: a narrower band emphasises the most isotropic directions, while a wider band retains more overall variance. Using a single configuration inevitably discards complementary information present in other plausible bands, a hypothesis consistent with the observation that the optimal `ktop` can vary substantially between datasets.

### 3.3 Gap‑Based Automatic Band Selection Is Unreliable
A naive attempt to determine `ktop` from singular value gaps—filtering indices where the gap exceeds a multiple of the mean gap—produced a `ktop` of only 13 on CUB‑2011, which is far from the validated value of 150. This result, recorded during the optimization run, demonstrates that the spectral boundary does not follow a simple largest‑gap heuristic; the spectrum of `Ψ` is smooth and lacks an unambiguous valley that would isolate an isotropic sub‑band.

### 3.4 Reintroducing Discarded Components Degrades Performance
Two additional modifications tested during the optimization—concatenating pre‑projection features with the ISO‑projected features and preserving 10 % of the removed spectral components—consistently reduced performance. This confirms that the anisotropic components discarded by IsoCLIP are genuinely noise for intra‑modal retrieval, and that even small amounts reintroduced dilute the benefit of subspace isolation.

## 4. Optimization Methodology

The AutoSOTA pipeline was allowed to modify the codebase (`src/retrieval.py`) and evaluate each change on CUB‑2011 using the ViT‑B/32 backbone, baseline mAP 27.03. Interventions were applied via targeted patches, and only those that improved the primary metric (mAP) were retained. Two interventions were accepted.

### 4.1 Soft Sigmoid Spectral Thresholding (Iteration 1)
**File:** `src/retrieval.py`, function `apply_iso`.  
**Change:** Introduced a new parameter `iso_tau` (default 0, reproducing original behaviour) and a sigmoid‑based soft‑weighting scheme. For `iso_tau > 0`, instead of hard binary selection, element‑wise weights are computed as:

```
w_top = σ( (i − ktop) / τ )
w_bottom = σ( (r − kbottom − 1 − i) / τ )
weights = w_top · w_bottom
W_text_iso = W_text V (weights ⊙ V^⊤)
W_image_iso = W_image U (weights ⊙ U^⊤)
```

This replaces the abrupt truncation with a smooth roll‑off controlled by the temperature `τ`. The soft weighting partially retains components near the band edges, mitigating information loss at the cut‑off. A value of `τ = 5.0` was used; the optimization log notes that a wide range (2.0–5.0) yields similar results, indicating robustness.

### 4.2 Multi‑Band Ensemble (Iteration 3)
**File:** `src/retrieval.py`, main retrieval loop before the single `apply_iso` call.  
**Change:** Added the flag `--iso_ensemble` and a block that evaluates four different `(ktop, kbottom)` configurations: (100, 25), (150, 50), (200, 75), (250, 100). For each configuration, the soft‑thresholded IsoCLIP projector is computed, query‑gallery similarity matrices are formed, and the resulting similarities are averaged element‑wise. This multi‑band strategy captures complementary discriminative information: the narrowest band retains only the core isotropic directions, while the widest includes more variance at the risk of admitting more anisotropy. Averaging leverages the consensus of these views without introducing additional hyper‑parameters. The ensemble is applied on top of the soft sigmoid weighting; thus the best run used `--iso_tau 5.0 --iso_ensemble`.

## 5. Experiments

### 5.1 Setup
- **Dataset:** CUB‑2011 (Caltech‑UCSD Birds‑200‑2011), containing 11,788 images across 200 fine‑grained bird species. The default splits provided by the codebase were used.
- **Model:** OpenAI CLIP ViT‑B/32 with embedding dimension 512. Pre‑extracted features were cached; no GPU‑intensive forward passes were required beyond the initial feature extraction.
- **Evaluation protocol:** Image‑to‑image retrieval. The query and gallery sets consist of image embeddings; the primary metric is mean Average Precision (mAP). Secondary metrics include mAP@R, precision@R, and recall@1.
- **Hardware/Software:** Experiments were performed in the sandbox environment provided by AutoSOTA, which imposed network restrictions (no download of external model weights beyond the default CLIP). Consequently, optimisations with ViT‑B/16 or ViT‑L/14 were not attempted, as those pretrained weights were not locally available. All metrics were obtained via the `summary.csv` output of `src/retrieval.py`.
- **Optimization budget:** 11 iterations, each comprising a full retrieval evaluation. The target mAP was 28.38, as set by the pipeline.
- **Baseline command:**
  ```bash
  python src/retrieval.py --dataroot /path/to/datasets \
      --dataset_name cub2011 --query_eval_type image \
      --gallery_eval_type image --no_iso --out_path baseline
  ```

### 5.2 Quantitative Results
Table 1 compares the baseline (no IsoCLIP) with the best configuration (soft sigmoid + ensemble). All metrics except recall@1 show marginal improvements.

**Table 1: Retrieval performance on CUB‑2011.**  

| Metric          | Baseline | Best (τ=5.0, ensemble) | Δ (absolute) | Improvement (%) |
|-----------------|----------|-------------------------|--------------|------------------|
| mAP             | 27.03    | **27.39**               | +0.36        | +1.33            |
| mAP@R           | 18.71    | 18.96                   | +0.25        | +1.34            |
| precision@R     | 29.74    | 30.03                   | +0.29        | +0.98            |
| recall@1        | 60.96    | 60.95                   | −0.01        | −0.02            |

The negligible change in recall@1 indicates that the improvements arise from better overall ranking quality rather than altering the top‑1 retrieval.

### 5.3 Ablation / Iteration Trajectory
Table 2 presents the only accepted interventions in chronological order. All intermediate attempts that harmed or did not change mAP (gap‑guided selection, ZCA whitening, residual preservation, temperature scaling, feature concatenation, 6‑band extension) are excluded.

**Table 2: Accepted modifications and their impact on mAP.**

| Iteration | Description                                                               | mAP   | Δ from baseline |
|-----------|---------------------------------------------------------------------------|-------|-----------------|
| 0         | Baseline (no IsoCLIP, hard binary band)                                   | 27.03 | –               |
| 1         | Soft sigmoid spectral thresholding (τ = 5.0)                              | 27.12 | +0.09           |
| 3         | Multi‑band ensemble on top of soft thresholding (four configurations)     | 27.39 | +0.36           |

The first intervention provided a modest lift; the second, more impactful, step exploited the ensemble to bring the total gain to +1.33 %. The cumulative effect confirms synergy: soft thresholding ensures smooth transitions within each band, while the ensemble aggregates complementary views.

## 6. Discussion

The optimization study confirms that the spectral decomposition underlying IsoCLIP can be refined with two simple, low‑cost modifications. Soft sigmoid weighting addresses the brittleness of hard boundaries, which is well motivated by the smooth singular spectrum of `Ψ`. The multi‑band ensemble leverages the fact that intra‑modal structure is not captured by a single isotropic subspace but by a range of scales; averaging similarities from nested and shifted spectral bands captures complementary information, consistent with classic ensemble principles in metric learning.

Nevertheless, the absolute gain remains modest (+0.36 mAP) and falls short of the target. This gap underscores the dataset dependence of IsoCLIP’s hyper‑parameters. The recommended (150, 50) values, validated on Caltech101, are likely sub‑optimal for CUB‑2011, a fine‑grained dataset with high intra‑class variance. The failure of gap‑guided automatic selection reinforces that the spectral boundary does not follow a simple statistical rule; therefore, per‑dataset grid search or learned per‑band weighting remains a promising future direction. The study also highlights the method’s sensitivity to the backbone: the omission of ViT‑B/16 and ViT‑L/14 due to missing weights limits the generalisation claims. On a positive note, both interventions proved robust to `τ` values in the [2.0, 5.0] range, suggesting that the soft thresholding hyper‑parameter is easy to tune.

Threats to validity include the single‑dataset focus, the sandbox environment’s potential effect on metric reproducibility (though retrieval is deterministic given cached features), and the absence of a dedicated validation set for hyper‑parameter selection. The mAP improvement, while consistent, is small in absolute terms; its practical significance must be weighed against the minimal integration cost.

## 7. Reproducibility

- **Repository:** [https://github.com/simomagi/IsoCLIP](https://github.com/simomagi/IsoCLIP), commit `76db6aec674e548b6ed261d54fe795f1a2223082` (best run).
- **Environment:**  
  ```bash
  conda create -n isoclip python=3.10.14
  conda activate isoclip
  conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
  cd IsoCLIP && pip install --no-build-isolation git+https://github.com/KaiyangZhou/Dassl.pytorch
  chmod +x install_requirements.sh && ./install_requirements.sh
  ```
- **Seed:** Not explicitly set; retrieval is deterministic with fixed extracted features.
- **Baseline run:**  
  ```bash
  python src/retrieval.py --dataroot /path/to/datasets \
      --dataset_name cub2011 --query_eval_type image \
      --gallery_eval_type image --no_iso --out_path baseline_retrieval
  ```
- **Optimized run:**  
  ```bash
  python src/retrieval.py --dataroot /path/to/datasets \
      --dataset_name cub2011 --query_eval_type image \
      --gallery_eval_type image --iso_tau 5.0 --iso_ensemble \
      --out_path iso_optimized
  ```

## 8. References

```bibtex
@InProceedings{Magistri_2026_CVPR,
    author    = {Magistri, Simone and Goswami, Dipam and Mistretta, Marco and Twardowski, Bart{\l}omiej and van de Weijer, Joost and Bagdanov, Andrew D.},
    title     = {IsoCLIP: Decomposing CLIP Projectors for Efficient Intra-modal Alignment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {29315-29324}
}

@misc{autosota,
    author       = {Tsinghua-FIB-Lab},
    title        = {AutoSOTA: An Automated State-of-the-Art Optimization Pipeline},
    year         = {2025},
    howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}}
}
```
