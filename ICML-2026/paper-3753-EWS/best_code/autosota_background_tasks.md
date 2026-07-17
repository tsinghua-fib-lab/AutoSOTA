# Background Task Ledger

## Task 1: Iteration 1 Training (GN + DropPath 0.2)
- **Task ID**: iter1_gn_droppath
- **Idea ID**: iter1
- **Command**: python train.py configs/ewsegnet/uper_zerowaste_40k_iter1.py --gpus 2 --work-dir work_dirs/iter1_gn_droppath --seed 42
- **Work dir**: /repo/work_dirs/iter1_gn_droppath
- **Log path**: /repo/work_dirs/iter1_gn_droppath/train.log
- **PID**: TBD
- **Start time**: 2026-07-16 22:14 UTC
- **Deadline**: 2026-07-17 02:14 UTC (4 hours)
- **Expected output**: Checkpoint at work_dirs/iter1_gn_droppath/latest.pth, validation mIoU logged every 4000 iters
- **Config changes**: GN (num_groups=32) replacing BN in decode_head + auxiliary_head; DropPath rate 0.2 (up from 0.1)
- **Score row**: iter=1, idea_id=iter1_gn_droppath
