# COGNOS Code Analysis — Paper 170

## Overview
COGNOS (Constrained Gaussian-Noise Optimization and Smoothing) is a universal time series anomaly detection framework combining KAN-AD with Gaussian regularization and Kalman smoothing.

## Evaluation Path
- Entry: run.py -> Exp_Anomaly_Detection.train() -> .test()
- Training: Forward pass -> MSE/GWNRLoss -> backprop -> Adam optimizer
- Inference: Model forward -> residual computation -> Kalman filtering -> SPOT threshold -> metrics
- Metrics parsed: F1, Aff-F1, VUS-ROC, VUS-PR, R-AUC-ROC, R-AUC-PR

## Key Files
- models/KANAD.py: Model architecture (BatchNorm, Conv1D, cosine basis)
- exp/exp_anomaly_detection.py: Training loop, evaluation, metric assembly
- utils/post_processing.py: Kalman filter, residual processing
- utils/tools.py: LR scheduling, EarlyStopping
- utils/losses.py: GWNRLoss with MMD + spectral flatness
- utils/metrics.py: adjusted_metric, affiliation_metric, auc_vus_metric

## Safe Modification Targets
- models/KANAD.py: Architecture changes
- utils/post_processing.py: Numerical fixes, adaptive smoothing
- utils/tools.py: LR schedules, EarlyStopping
- exp/exp_anomaly_detection.py: Training loop improvements

## Risky Files (DO NOT MODIFY)
- utils/metrics.py: Metric computation
- affiliation/: Affiliation metric library
- data_provider/: Data loading and splits
- /datasets/MSL/: Test data and labels

## Baseline Metrics
Std-F1=0.9164, Aff-F1=0.915, R-A-R=0.6674, R-A-P=0.2098, V-R=0.648, V-P=0.1765
