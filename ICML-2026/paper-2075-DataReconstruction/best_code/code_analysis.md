# Code Analysis for Paper 2075: Data Reconstruction with Sample Splitting

## Evaluation Path
- **Entry point**: `Main.py:main()` → `main_reconstruct()` → `improved_data_extraction()`
- **Metric parsing**: stdout line containing `Extraction-Score=<L2_dist>` printed in `extraction.py:evaluate_extraction()`
- **Metric**: Top-10 mean L2 distance between reconstructed and training samples (lower is better)
- **Baseline**: L2_Distance = 4.72

## Config Path
- `GetParams.py` - CLI argument definitions
- `settings.py` - paths (datasets_dir=/datasets, models_dir=/models, results_base_dir=/autosota_cache/results)
- `command_line_args/` - preset arg files

## Key Files
| File | Purpose | Risk Level |
|------|---------|------------|
| `Main.py` | Entry point, train/reconstruct orchestration | MEDIUM - contains lr_decay bug |
| `extraction.py` | Extraction loss, training, evaluation | MEDIUM - commented KKT scaling |
| `split.py` | Sample splitting (eigenvalue + Lanczos) | HIGH - computational bottleneck |
| `evaluations.py` | L2 distance, DSSIM, PSNR scoring | HIGH - do not modify metric calc |
| `GetParams.py` | CLI argument parsing | LOW - safe to add args |
| `CreateModel.py` | Model architecture creation | LOW |

## Risky Files (DO NOT MODIFY)
- `evaluations.py:l2_dist()`, `viz_nns()`, `get_evaluation_score_dssim()` - metric computation
- `common_utils/datasets.py` - dataset loading/splitting

## Safe Modification Targets
- `Main.py:407` - lr_decay=False hardcode to lr_decay=args.lr_decay
- `Main.py:239` - splitting condition (extraction_epochs-based)
- `Main.py:248` - epsilon parameter (0.02 passed to sample_splitting)
- `Main.py:284` - torch.cuda.empty_cache() frequency
- `extraction.py:30` - initialization scale and method
- `extraction.py:44-57` - optimizer and LR scheduler
- `extraction.py:159-160` - commented KKT loss scaling
- `extraction.py:121-130` - verify loss weights
- `GetParams.py` - new CLI arguments (safe to add)
- `split.py:233-297` - sample_splitting epsilon optimization

## Pre-trained Weights
- `/models/weights-mnist_odd_even_d50_mnist_odd_even.pth` - trained model (200K epochs)
- `/models/weights-mnist_odd_even_d50_mnist_odd_even_initial.pth` - initial model

## Baseline Evaluation Command
```
cd /repo && WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python3 Main.py --run_mode=reconstruct --extraction_method=Loo --problem=mnist_odd_even --data_per_class_train=50 --extraction_data_amount_per_class=100 --extraction_epochs=300000 --extraction_evaluate_rate=1000 --max_extraction_iter=50000 --model_hidden_list=[1000,1000] --model_init_list=[0.001,0.001] --pretrained_model_path=weights-mnist_odd_even_d50_mnist_odd_even.pth --initial_model_path=weights-mnist_odd_even_d50_mnist_odd_even_initial.pth --wandb_active=False
```

## Known Bugs
1. `Main.py:407` - `lr_decay=False` hardcoded, ignores `--lr_decay` CLI arg
2. `extraction.py:159-160` - KKT loss scaling factor (200x) commented out for Loo method
3. `Main.py:284` - `torch.cuda.empty_cache()` called every epoch (unnecessary overhead)

## Paper Mechanism
- **Sample Splitting**: When `extraction_epochs < max_extraction_iter`, periodic splitting of reconstruction atoms along negative curvature directions
- **Splitting condition**: `epoch % args.extraction_epochs == 0 and epoch > 0 and epoch < max_iter` (Main.py:239)
- **Growth rate**: 0.3 when fewer than 1000 atoms, 0.1 when 1000+ atoms
- **Epsilon**: 0.02 (perturbation magnitude along negative curvature)
