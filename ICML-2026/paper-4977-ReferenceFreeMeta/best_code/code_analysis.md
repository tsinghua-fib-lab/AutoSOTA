# Code Analysis for Paper 4977 (RFML4MRI - DINER)

## Evaluation Path
- **Entry**: `eval_diner.py` — test-time adaptation of DINER INR on fastMRI slices
- **Args**: `--data_dir`, `--num_slices`, `--steps`, `--lr`, `--tv_weight`, `--step_size`, `--gamma`, `--checkpoint`, `--output_dir`
- **Flow**: Load samples → For each slice: create DinerModel → AdamW optimizer + StepLR scheduler → 300-step adaptation (L1 k-space loss + TV reg) → compute PSNR
- **Output**: JSON at `<output_dir>/eval_results.json` with psnr_mean, psnr_std, psnr_min, psnr_max, psnr_median, individual_psnrs, config
- **Metrics**: PSNR computed on ROI (masked by undersampling mask) using `calculate_psnr()` (normalize + 10*log10)

## Training Path
- **Entry**: `diner_ipod_train.py` — meta-training with Reptile + DINER backbone
- **Class**: `DinerReptileTrainer` with `inner_loop_adaptation()` and `adaptive_reptile_update()`
- **Hyperparams**: inner_lr=2e-2, meta_lr=5e-4, inner_steps=300, epochs=2500, tasks_per_epoch=15, samples_per_task=5
- **Checkpoints**: Saved every 100 epochs; best model kept at `/repo/checkpoints_diner/best_model.pth`
- **Data**: `/datasets/fastmri_processed` (40 train + 15 eval tasks, AF=10 Cartesian 1D)

## Config Path
- Eval config: CLI args in `eval_diner.py` (no YAML/JSON config file)
- Training config: Inline Python dict at top of `diner_ipod_train.py` main()
- Model config: `encoding_config` and `network_config` dicts in `model_diner.py` DinerModel.__init__()

## Model Architecture
- `HashEncoding`: 16-level multi-resolution hash grid, 2 features/level, hash table size 2^19
- `DinerMLP`: 2 hidden layers, 16 neurons, ReLU, separate mag/phase branches
- Shared hash encoding, separate MLP for magnitude and phase

## Safe Modification Targets (for eval-only improvements)
1. `eval_diner.py::test_time_adaptation()` — loss function, optimizer config, LR schedule, step logic
2. `eval_diner.py::main()` — CLI args, default values

## Safe Modification Targets (for meta-training improvements)
3. `diner_ipod_train.py::DinerReptileTrainer::inner_loop_adaptation()` — inner loop optimization
4. `diner_ipod_train.py::DinerReptileTrainer::adaptive_reptile_update()` — meta-update logic
5. `diner_ipod_train.py::DinerReptileTrainer.train()` — training loop
6. `model_diner.py::DinerModel`, `HashEncoding`, `DinerMLP` — architecture changes (need retrain)

## Risky Files (do not modify)
- `utils.py::calculate_psnr()` — metric computation
- `utils.py::normalize01()` — normalization used in PSNR calc
- Data loading functions — must preserve AF=10 Cartesian filter
- `/tools/record_score.sh` — score recording

## Available Resources
- 2x NVIDIA A100-SXM4-80GB (GPU 0: 36GB used, GPU 1: 0GB used)
- 50 pre-processed eval slices at `/datasets/fastmri_eval_v2` (AF=10 Cartesian 1D)
- 40 train + 15 eval tasks at `/datasets/fastmri_processed`
- IPOD checkpoint at `/repo/checkpoints_diner/best_model.pth` (20 epochs)

## Baseline Results
- Random init (no IPOD): PSNR mean=40.26 dB, median=41.21 dB
- IPOD 20-epoch checkpoint: PSNR mean=40.53 dB, median=42.22 dB
- Paper reports 39.69 dB for multicoil brain with full training
