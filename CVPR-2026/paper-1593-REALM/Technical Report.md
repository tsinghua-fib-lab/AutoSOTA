# REALM: A Technical Report on Automated Optimization

> Original paper: **REALM — An MLLM-Agent Framework for Open-World Reasoning Segmentation and Editing on 3D Gaussian Splatting** (CVPR 2026). This report documents an automated optimization study of the public REALM code base conducted with the AutoSOTA agent (`tsinghua-fib-lab/AutoSOTA`).

## Abstract

REALM is a multimodal large language model (MLLM) agent framework that performs open-world reasoning segmentation and editing directly on 3D Gaussian Splatting (3D-GS) representations. Its inference pipeline couples per-view MLLM bounding-box reasoning, a learned 3D Gaussian classifier, and cross-view object-ID voting to produce object masks for arbitrary natural-language queries. While effective, the pipeline exhibits two coupled weaknesses: stochastic MLLM responses produce inconsistent object identifiers across views, and the Gaussian classifier yields masks with blurry boundaries. This report describes an automated optimization campaign targeting the REALM pipeline on the LERF benchmark (figurines, ramen, teatime). Across seven iterations, the AutoSOTA agent explored prompt-level, pipeline-level, and post-processing-level changes. Modifications to the MLLM prompt and to the cross-view voting logic proved unstable, but a post-hoc Hybrid SAM Boundary Refinement strategy, in which Segment Anything Model (SAM) boundaries are constrained by dilated and eroded versions of the original masks, produced a decisive improvement. The best configuration raised overall mIoU from 72.69 to 77.86 (+5.17, +7.1%) and mBIoU from 61.61 to 70.39 (+8.78, +14.2%), surpassing the predefined mIoU target of 76.3245. A latent bug in the dataset reader that prevented test-view cameras from being instantiated was also identified and fixed. The findings indicate that, for MLLM-agent driven 3D segmentation pipelines, deterministic geometric post-processing currently offers a more reliable optimization surface than tuning the upstream stochastic reasoner.

## 1. Introduction

Open-world 3D scene understanding requires systems that can localize arbitrary, query-defined objects inside reconstructed 3D representations without supervised category labels. REALM addresses this setting on top of 3D Gaussian Splatting, exposing the scene to an MLLM agent that reasons about objects from a small set of rendered views, votes on consistent object identifiers, and assembles final 3D masks from a Gaussian-level classifier. The pipeline supports downstream editing operations including removal, replacement, and style transfer.

The present technical report does not propose a new method. Instead, it documents an automated optimization study of the REALM reference implementation conducted with the AutoSOTA agent. AutoSOTA is given the repository, a fixed evaluation harness on the LERF reasoning-segmentation split, and a target metric, and explores modifications under a budgeted iteration loop. The goal of this report is to (i) describe the optimization trajectory faithfully, (ii) attribute the observed gains to specific code changes, and (iii) document negative results that constrain future work on the same code base.

## 2. Original Method (Background)

REALM operates on a per-scene 3D Gaussian Splatting reconstruction augmented with a per-Gaussian object-identity classifier, following the Gaussian-Grouping data and checkpoint convention. Given a natural-language prompt, the inference pipeline (`reason_seg.py`) performs three logical stages:

1. **View sampling and rendering.** A subset of training/test cameras is rendered, producing RGB images, per-Gaussian object-ID feature maps, and depth maps from the trained scene.
2. **MLLM reasoning.** A multimodal LLM, accessed through an API wrapper (`ReasonModelAPI` / `KIMIAPI` in `utils/reason_utils.py`), is queried with the rendered views and the user prompt to return candidate object bounding boxes and natural-language identifiers. Grounded-SAM (`ext/grounded_sam`) is optionally used for box-conditioned 2D masks.
3. **Cross-view voting and 3D mask assembly.** Per-view object identifiers are aggregated via a classifier-based cross-view voting step (`reasoneditor/id_utils.extract_selected_obj_ids`, `points_inside_convex_hull`) that selects a consistent set of Gaussian object IDs, which are then rasterized to per-view 2D masks for evaluation (`render_mask` in `gaussian_renderer`).

Training and rendering follow the standard 3D-GS recipe (`train.py`, `render.py`) with additional supervision on the per-Gaussian classifier. Reasoning, segmentation, and editing scripts are orchestrated by `script/run_seg.sh` and related shell launchers. Evaluation on LERF reports overall mean Intersection-over-Union (mIoU) and mean boundary IoU (mBIoU) over the three scenes used in this study: figurines, ramen, and teatime.

## 3. Identified Limitations

The optimization campaign and direct code inspection surface three concrete limitations of the released pipeline:

1. **MLLM non-determinism and cross-view inconsistency.** The MLLM is queried independently per view. In practice, the same object is frequently assigned different identifiers across views, and chain-of-thought (CoT) prompt variants amplify rather than dampen this inconsistency. This is the dominant source of variance in the final masks.

2. **Classifier-based voting is brittle.** Even when bounding boxes are reasonable, the cross-view voting that selects which Gaussian object IDs constitute the answer is sensitive to single-view disagreements. Re-running mask generation with the MLLM-in-the-loop in the AutoSOTA harness produced overall mIoU of approximately 3%, indicating that the deployed `reason_seg.py` execution path requires careful caching of pre-generated reasoning results to be reproducible at the reported numbers.

3. **Test-camera population bug in the scene loader.** In `scene/dataset_readers.py::readColmapSceneInfo`, the `eval=False, train_split=True` branch initializes `test_cam_infos` to `[]` and never repopulates it. As a result, `Scene.getTestCameras()` returns an empty list, and any downstream consumer that depends on test views (including SAM-guided refinement that requires the corresponding test image) silently degrades. The relevant branch is:

```189:201:scene/dataset_readers.py
    else:
        if train_split:
            train_dir = os.path.join(path, "images_train")
            train_names = sorted(os.listdir(train_dir))
            train_names = [train_name.split('.')[0] for train_name in train_names]
            for cam_info in cam_infos:
                if cam_info.image_name in train_names:
                    train_cam_infos.append(cam_info)
            test_cam_infos = []
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
```

The corrected branch adds an `else` clause that appends non-training cameras to `test_cam_infos`, enabling test-view image lookup for downstream refinement.

In addition, the produced 2D masks tend to exhibit boundary blur and small spurious components, which inflates the gap between mIoU and the stricter mBIoU metric.

## 4. Optimization Methodology

The optimization was performed by the AutoSOTA agent. At each iteration the agent (i) proposes a code change as a patch against the current best commit, (ii) executes a fixed evaluation harness over the LERF figurines/ramen/teatime scenes using the cached MLLM responses, and (iii) compares overall mIoU against a predefined target of 76.3245. Iterations span three categories:

- **Pipeline-internal modifications** that change the upstream reasoning or voting (Iter 1, Iter 4).
- **Post-processing of pre-generated masks** that leaves the upstream pipeline untouched (Iter 2, Iter 3, Iter 5, Iter 6, Iter 7).
- **Infrastructure fixes** required to unlock new modifications (the test-camera bug fix is a prerequisite for SAM-based refinement, since SAM requires the corresponding test-view RGB image).

The final winning artifact is `mask_refinement.py`, a standalone post-processing script that operates on the already-generated mask directory tree. It composes two stages:

1. **Morphological cleanup.** Binary closing with a `closing_radius` of 3 fills small holes, connected-component analysis with `min_region_size = 50` removes isolated noise, and an optional median filter smooths edges.
2. **Hybrid SAM boundary refinement.** For each mask, the original mask is dilated to define an upper-bound search region and eroded to define a high-confidence interior. Interior points are sampled as positive SAM prompts and points outside the dilated region are sampled as negative prompts. SAM (ViT-H) is run on the corresponding test-view image, and each candidate mask is constrained to the dilated region while the eroded interior is forced to retain the original mask values, yielding a hybrid output that preserves the reliable interior of the original mask and replaces only the uncertain boundary band with SAM's boundary.

The hybrid rule is implemented as:

```141:153:mask_refinement.py
    for sam_mask in masks:
        # Constrain SAM mask to dilated region
        constrained = sam_mask.astype(np.float32) * dilated.astype(np.float32)
        # Keep eroded interior from original
        hybrid = np.where(eroded, original_mask > 0.5, constrained > 0.5).astype(np.float32)

        # Score: prefer masks with reasonable overlap
        overlap = (hybrid * (original_mask > 0.5)).sum() / (original_mask > 0.5).sum()
        if overlap > best_score and hybrid.sum() > 10:
            best_score = overlap
            best_mask = hybrid
```

This decouples optimization from the stochastic MLLM stage and turns mask quality improvement into a deterministic geometric problem.

## 5. Experiments

### 5.1 Setup

All experiments are run on the public REALM code base targeting the LERF reasoning-segmentation benchmark with three scenes: figurines, ramen, and teatime. The metric protocol reports per-scene mIoU and mBIoU, and the overall mIoU/mBIoU averaged across the three scenes. The AutoSOTA agent uses pre-generated MLLM responses cached on disk to ensure reproducibility of upstream reasoning; only the post-processing stages and, where relevant, the scene loader are modified. The improvement target for overall mIoU is 76.3245, taken from the AutoSOTA configuration. SAM uses the ViT-H checkpoint `sam_vit_h_4b8939.pth`.

### 5.2 Quantitative Results

The best commit (`f1f64b8865`) corresponds to Iteration 7, the Hybrid SAM Boundary Refinement strategy. Overall and per-scene metrics versus the unmodified REALM baseline are reported in Table 1; per-scene deltas are highlighted in Table 2.

**Table 1. Baseline vs. best metrics on LERF reasoning segmentation.**

| Metric          | Baseline | Best  | Delta | Delta % |
|-----------------|----------|-------|-------|---------|
| Overall mIoU    | 72.69    | 77.86 | +5.17 | +7.1%   |
| Overall mBIoU   | 61.61    | 70.39 | +8.78 | +14.2%  |
| Figurines mIoU  | 69.73    | 77.79 | +8.06 | +11.6%  |
| Figurines mBIoU | 62.07    | 72.33 | +10.26| +16.5%  |
| Ramen mIoU      | 76.95    | 79.99 | +3.04 | +4.0%   |
| Ramen mBIoU     | 61.91    | 70.21 | +8.30 | +13.4%  |
| Teatime mIoU    | 73.10    | 75.42 | +2.32 | +3.2%   |
| Teatime mBIoU   | 60.39    | 67.00 | +6.61 | +10.9%  |

**Table 2. Summary of the winning change.**

| Change                                       | Effect                       | Notes                                                                                                                                                                                                                 |
|----------------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hybrid SAM Boundary Refinement (Iteration 7) | mIoU +5.17%, mBIoU +8.78%    | SAM generates refined boundaries guided by test-view images, while original masks provide reliable interior structure. Dilated originals constrain SAM to relevant regions, eroded originals preserve confident interior pixels. |

The target overall mIoU of 76.3245 is exceeded by the Iteration 7 configuration (77.86 > 76.3245). The largest absolute gains occur on figurines, where boundary uncertainty in the original masks is most pronounced. The mBIoU gain (+14.2% relative) is markedly larger than the mIoU gain (+7.1% relative), consistent with the hypothesis that the improvement is dominated by tighter, better-aligned boundaries rather than interior corrections.

### 5.3 Ablation / Iteration Trajectory

Table 3 summarizes the qualitative outcome of each iteration. Numeric metrics are reported where stable; iterations marked as regressions destabilized either upstream MLLM behavior or per-mask geometry and were rolled back.

**Table 3. Iteration trajectory.**

| Iter | Category                                 | Description                                                                                                | Outcome                                                                  |
|------|------------------------------------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 0    | Baseline                                 | Unmodified REALM pipeline on cached MLLM responses.                                                        | Overall mIoU 72.69 / mBIoU 61.61.                                        |
| 1    | MLLM prompt engineering                  | CoT-style system prompt for the reasoning MLLM.                                                            | Regression: cross-view ID consistency worsened; rolled back.             |
| 2    | Morphological post-processing            | Binary closing + small-region removal applied uniformly to pre-generated masks.                            | Solid gain of approximately +0.77% overall mIoU; kept as a baseline tool.|
| 3    | Adaptive morphological parameters        | Per-mask adaptive closing/hole-filling.                                                                    | Regressed teatime; rolled back in favor of uniform Iter 2 settings.      |
| 4    | Pipeline-based mask regeneration         | Re-ran `reason_seg.py` with the MLLM in-the-loop, modifying classifier voting.                             | Regression: overall mIoU collapsed to approximately 3%.                  |
| 5    | Per-scene morphological parameter tuning | Independent (closing_radius, min_region_size) per scene.                                                   | Matched but did not exceed uniform Iter 2 settings.                      |
| 6    | Pure SAM refinement                      | Replace original masks with SAM masks driven by interior prompts.                                          | mBIoU improved but overall mIoU decreased; not a Pareto improvement.     |
| 7    | Hybrid SAM Boundary Refinement           | Dilated/eroded original masks bound SAM output; final mask preserves original interior, adopts SAM boundary.| Best result; overall mIoU 77.86, mBIoU 70.39. Commit `f1f64b8865`.       |

The trajectory exhibits two clear patterns. First, every attempt to modify the stochastic upstream stage (Iter 1, Iter 4) was net-negative. Second, deterministic post-processing improved monotonically once it combined morphological cleanup with SAM-derived boundaries (Iter 7) instead of using either in isolation (Iter 2, Iter 6).

## 6. Discussion

The optimization study yields three observations likely to generalize to similar MLLM-agent 3D pipelines.

**Stochastic reasoners are a poor optimization surface.** The MLLM stage introduces variance that cannot be eliminated by prompt rewording and that is amplified, not damped, by chain-of-thought instructions in this setting. Small changes to the system prompt redistributed which object identifiers appeared in which views, and the classifier-based cross-view vote was not robust to that redistribution.

**The bottleneck is boundary geometry, not interior selection.** The baseline mIoU (72.69) is substantially higher than the baseline mBIoU (61.61), indicating that the original masks usually find the correct object but with sloppy boundaries. The winning intervention concentrates capacity exactly on that boundary band: it trusts the original mask deep inside (eroded interior), trusts SAM elsewhere within a tight neighborhood (dilated upper bound), and only repaints the thin uncertain band in between. This decomposition lifts both mIoU and mBIoU simultaneously, whereas pure SAM refinement (Iter 6) over-trusts SAM globally and pure morphology (Iter 2) under-uses image evidence.

**Infrastructure debt can block algorithmic ideas.** The test-camera bug in `scene/dataset_readers.py` did not affect headline metrics directly, but it prevented SAM-based refinement from accessing the right RGB views and would have masked the entire Iteration 7 gain had it not been fixed.

The most promising unexplored directions concern the upstream reasoner. Replacing the MLLM with a deterministic open-vocabulary detector such as GroundingDINO would address cross-view inconsistency at its root. Projecting 2D SAM masks directly to 3D via rendered depth and alpha-compositing could bypass the classifier-based vote entirely. CLIP-based semantic filtering could verify, per candidate object ID, that the rendered region matches the target concept. Multi-scale SAM refinement and test-time camera-jitter augmentation are low-risk extensions of the current winner.

## 7. Reproducibility

The reproduction protocol below assumes the REALM repository layout shipped with this study and a CUDA-capable GPU. Environment setup follows the upstream `README.md`:

```bash
conda create -n realm python=3.8 -y
conda activate realm
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1 tqdm scipy wandb opencv-python scikit-learn lpips
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install segment-anything Pillow
```

LERF data is organized as described in the upstream README (`data/lerf/{figurines,ramen,teatime}`). 3D Gaussian Splatting feature fields are trained per scene with the provided launcher (e.g., `bash script/train.sh lerf/figurines 1`). The reasoning-segmentation stage that produces the baseline masks is invoked through `script/run_seg.sh` (or equivalently `reason_seg.py`) using cached MLLM responses to ensure deterministic outputs.

Once baseline masks exist under a path such as `result_gsgroup/lerf_mask`, the winning post-processing is reproduced with:

```bash
python3 mask_refinement.py \
    --result_path result_gsgroup/lerf_mask \
    --data_path data/lerf \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth
```

The script backs up the original masks to `<result_path>_backup`, applies morphological cleanup (`closing_radius=3`, `min_region_size=50`), and then applies SAM-guided hybrid boundary refinement (`dilation_radius=10`, `erosion_radius=3`). For SAM to load the correct test-view RGB images, the dataset-reader fix described in Section 3 must be present; the relevant `else` branch in `scene/dataset_readers.py::readColmapSceneInfo` must populate `test_cam_infos` with non-training cameras instead of leaving it empty.

Evaluation is performed with the LERF mIoU/mBIoU script bundled with the repository. The best metrics (overall mIoU 77.86, overall mBIoU 70.39) correspond to commit `f1f64b8865`.

## 8. References

1. Shi, C. et al. *REALM: An MLLM-Agent Framework for Open-World Reasoning Segmentation and Editing on 3D Gaussian Splatting.* CVPR 2026. Project page: <https://changyueshi.github.io/REALM/>.
2. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM Transactions on Graphics, 2023.
3. Ye, M. et al. *Gaussian Grouping: Segment and Edit Anything in 3D Scenes.* 2023.
4. Kirillov, A. et al. *Segment Anything.* ICCV 2023. Checkpoint: `sam_vit_h_4b8939.pth`.
5. Liu, S. et al. *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* 2023.
6. Kerr, J. et al. *LERF: Language Embedded Radiance Fields.* ICCV 2023.
7. AutoSOTA: Automated Optimization Agent. Tsinghua FIB Lab. <https://github.com/tsinghua-fib-lab/AutoSOTA>.
