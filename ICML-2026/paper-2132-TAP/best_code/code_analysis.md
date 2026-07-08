# Code Analysis - Paper 2132 (TAP)

## Evaluation Path
- Script: /repo/eval_miceprotein_v2.py
- Command: python3 eval_miceprotein_v2.py --synthetic_path runs_v3/MiceProtein_n20/synthetic_data.csv
- Metrics: Accuracy, Macro-F1 (6 classifiers x 5-fold CV)
- Output format: Accuracy: X.XXXX +/- X.XXXX, Macro-F1: X.XXXX +/- X.XXXX, JSON line

## Training Path
- Main loop: /repo/run_tap_v3.py (200 KTO steps)
- Generation: environment.py -> DataGenerationEnv.step() -> generator.sample_inpaint()
- Policy: kto_controller.py -> KTOAgent.update() (KTO loss)
- TabDiff: generators/tabdiff.py -> TabDiffGenerator.sample_inpaint()
- Diffusion: TabDiff/tabdiff/models/unified_ctime_diffusion.py:sample_inpaint()

## Config
- /repo/config.py: KTOConfig, InpaintConfig, TrainConfig

## Key Modification Targets
- unified_ctime_diffusion.py:sample_inpaint() - RePaint resampling (A-01)
- environment.py:_select_anchor_indices() - Anchor quality (A-03)
- environment.py:_diversity_gate() - Dynamic recalibration (C-01)
- environment.py:step() - Curriculum scheduling (A-05)
- config.py - Parameter sweeps (P-01)

## Baseline
- Accuracy: 50.42, Macro-F1: 47.38
- TabDiff: runs_v3/MiceProtein_n20/model/ (8000 steps, full dataset)
