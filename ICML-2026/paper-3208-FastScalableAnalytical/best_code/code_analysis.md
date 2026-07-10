# Code Analysis — GoldDiff (Paper 3208) SOTA Optimization

## Evaluation Path
- Entry: `main.py` → `run_sampling()` → `evaluate_main_model()` → `evaluate_comparison()`
- Metrics recorded in `motivation_figures_2/ours/<run_name>/metrics.json`
- Key fields: `mse_score`, `r2_score`, `main_sampling_time_per_step_est`
- Postprocessing: `dataset_bundle.postprocess()` normalizes output to [0,1]
- Metric functions: `calculate_mse()`, `calculate_r2_score()` in `services/evaluation.py`
- Baseline comparison: U-Net from `services/base_models/baseline_unet/cifar10/ckpt_epoch_200.pt`

## Training/Inference Path
- `methods/ours.py::OursDenoiser.train()` loads Wiener SVD, builds x0 database
- `methods/ours.py::OursDenoiser.denoise()` per-step:
  1. `_compute_projection_mask()` — D×D matmul to compute PCA locality mask
  2. `_get_knn_subset_and_distances()` — hierarchical KNN (coarse low-res → fine full-res)
  3. `_denoise_knn()` — einsum + streaming softmax aggregation

## Config Path
- Hierarchy: `configs/ours/<name>.yaml` → `defaults: [/defaults.yaml]` → structured `Config` dataclass
- CIFAR-10 config: `configs/ours/cifar10.yaml`
- Key params: `k_min=2500`, `k_max=5000`, `temperature=1.0`, `mask_threshold=0.03`, `chunk_size=1024`

## Metric Parser
- `services/evaluation.py::_log_metrics()` writes `metrics.json`
- Scheduler reads: `mse_score`, `r2_score`, `main_sampling_time_per_step_est`

## Reusable Resources
- `/repo/services/base_models/wiener/cifar10_32/` — cached Wiener SVD (U, LA, Vh, mean)
- `/repo/services/base_models/baseline_unet/cifar10/ckpt_epoch_200.pt` — DDPM U-Net checkpoint
- `/repo/data/CIFAR10/cifar-10-batches-py/` — CIFAR-10 dataset (50k training images)

## Pre-existing Dynamic Schedules
- `dynamic_m` and `dynamic_k` default to `True` in `_init_knn_params()` (base.py:332-333)
- They are NOT overridden in cifar10.yaml → already active
- Schedule: more data at high noise (early steps), less at low noise (late steps)
- Bounds: n_pre_sample ∈ [5000, 50000], k_final ∈ [2500, 5000]

## Safe Modification Targets
- `methods/ours.py` — mask computation, denoising aggregation
- `methods/base.py` — KNN retrieval, timestep scheduling, sampling loop
- `main.py` — seeding, device setup
- `configs/ours/cifar10.yaml` — hyperparameters
- `services/evaluation.py` — READ-ONLY: metric computation
- `services/wiener.py` — READ-ONLY: Wiener filter computation

## Risky Files (Do Not Modify)
- Any file under `services/third_party/edm/` — external dependency
- `data_src/datasets.py` — dataset loading
- `networks/UNet.py` — U-Net architecture
