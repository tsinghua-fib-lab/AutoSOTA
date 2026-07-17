# Final Report: paper-3753

- Title: Towards Effective Waste Segmentation for Automated Waste Recycling in Cluttered Background
- Primary metric: `mIoU` (higher)
- Records: 2
- Generated: 2026-07-16T22:26:47Z

## Best Result

- Iteration: 1
- Idea: iter1_ms_tta — Multi-Scale TTA (--aug-test) on baseline checkpoint
- Primary metric: 57.25
- Commit: `0b8b7ca5b218cd8055e2ae2d0b5d5956a61c4816`
- Notes: Idea 8: Multi-scale test-time augmentation using --aug-test flag (scales: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75 + flip). mIoU improved from 57.14 to 57.25 (+0.11 pp). Pixel Accuracy improved from 91.78 to 92.11 (+0.33 pp). No training cost. Baseline checkpoint used.
