# Background Task Ledger

## Completed Tasks

### Task 1: ALGO-01 Multi-Position Training (COMPLETED)
- **ID**: train-algo01
- **Iteration**: 2 (5K), 4 (10K)
- **Idea**: ALGO-01 (projection_positions: encoder+bottleneck+decoder)
- **Results**: 5K MSE=0.158 RMAE=0.050, 10K MSE=0.092 RMAE=0.040
- **Verdict**: Better RMAE but worse MSE than baseline (0.080). Trade-off.
- **Status**: completed (killed after 10K eval)

### Task 2: CODE-04 fd_acc=4 Training (COMPLETED)
- **ID**: train-code04
- **Iteration**: 1 (5K), 3 (10K)
- **Idea**: CODE-04 (fd_acc: 4)
- **Results**: 5K MSE=0.225 RMAE=0.071, 10K MSE=0.118 RMAE=0.046
- **Verdict**: Worse than baseline on primary metric. fd_acc=4 harmful.
- **Status**: completed (killed after 10K eval)

## Running Tasks

### Task 3: Combo (ALGO-01 + CODE-04)
- **ID**: train-combo-ac
- **Iteration**: 5
- **Idea**: ALGO-01 + CODE-04 combined (multi-pos + fd_acc=4)
- **Command**: python3 main.py --config configs/sota/model.combo_ac.yaml --gpu 0
- **Log**: /tmp/train_combo_ac.log
- **PID**: 6898
- **Start**: 2026-07-05 15:58 UTC
- **Deadline**: 2026-07-05 16:28 UTC
- **Status**: running

### Task 4: PARAM-02 c_projection=0.02
- **ID**: train-param02c
- **Iteration**: 6
- **Idea**: PARAM-02 (c_projection=0.02, lower physics weight)
- **Command**: python3 main.py --config configs/sota/model.param02_cproj002.yaml --gpu 1
- **Log**: /tmp/train_param02c.log
- **PID**: 6970
- **Start**: 2026-07-05 15:58 UTC
- **Deadline**: 2026-07-05 16:28 UTC
- **Status**: running
