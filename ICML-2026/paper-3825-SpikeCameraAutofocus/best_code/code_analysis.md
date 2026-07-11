# Code Analysis — Paper 3825: CEN Spike Camera Autofocus

## Evaluation Path
- Script: scripts/eval_sad_cen_v2.py
- Command: python3 scripts/eval_sad_cen_v2.py --sad_root /datasets/sad_dataset --output results_sad_cen.csv
- Dataset: SAD dataset at /datasets/sad_dataset (15 scenes)
- Output: CSV with per-scene AbsErr, RelErr, r2, plus MEAN summary row

## Core Algorithm
- cen.py: CENConfig, run_cen_curves(), score_curve_general(), pick_r2_gtfree(), estimate_focus_from_blocks()
- io.py: stream_blocks_from_dat(), stream_blocks_from_npy_files()
- Scoring weights: alpha_prom=1.10, alpha_curv=0.35, alpha_width=-0.45, alpha_plat=-0.35, alpha_edge=-0.60

## Key Observations
1. USAF scenes drive error (mean 5.67 vs Bottle 1.0)
2. r2 auto-selection is primary lever
3. No multimodality penalty in scoring
4. No windowing before FFT
5. scipy now installed for sg_filter and argrelmax

## Safe Targets: cen.py functions, CENConfig dataclass
## Risky: eval script metric parsing, config.xlsx labels, dataset files
