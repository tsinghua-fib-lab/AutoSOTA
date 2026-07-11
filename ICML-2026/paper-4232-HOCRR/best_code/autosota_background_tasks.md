# Background Task Ledger

## Task 1: Iter-2 Extended Training
- **Idea ID:** ALGO-01 + CODE-02
- **Command:** python3 -u experiments/mnist_rotation/train_e2cnn_rotation.py
- **Working directory:** /repo
- **Log:** /repo/training_iter2.log
- **PID:** $(cat /tmp/training_iter2.pid 2>/dev/null || echo "unknown")
- **Started:** 2026-07-11 04:48 UTC
- **Deadline:** 2026-07-11 06:48 UTC (2 hours)
- **Expected output:** models/e2cnn_rotation_model.pth
- **Score row:** iter=2, idea=ALGO-01+CODE-02
- **Status:** running

## Task 2: Iter-4 PARAM-01 n_trials=5 Estimation
- **Idea ID:** PARAM-01
- **Command:** mnist_rotation_full_certification.py --n_trials 5 --device cuda
- **Working directory:** /repo
- **Log:** /repo/estimation_iter4.log
- **GPU:** 1
- **Output:** outputs/mnist_sigma0p75_estimation_nt5.json
- **Started:** $(date -u)
- **Deadline:** +1 hour
- **Status:** running
