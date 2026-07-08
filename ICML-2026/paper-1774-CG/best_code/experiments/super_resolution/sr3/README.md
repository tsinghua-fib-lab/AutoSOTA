# SR3 1000-image test list

`sr3_top1k.txt` — the 1000-image ImageNet-val list used by the SR3 benchmark
(Pandey et al. 2025b), as `wnid/filename gt_class_id` per line. This is the test
set for the paper's Table 5 super-resolution numbers.

`sr3_top1k_predicted_labels.csv` — `filename,wnid,gt_class_id,pred_class_id,pred_top5`.
`pred_class_id` is the top-1 prediction of the noise-aware 64x64 ImageNet
classifier (Dhariwal & Nichol) on the bicubic LR input; the paper conditions the
(class-conditional) pMF prior on this prediction, since SR itself is not
class-conditional. Pass it via `--predicted-labels-csv`.
