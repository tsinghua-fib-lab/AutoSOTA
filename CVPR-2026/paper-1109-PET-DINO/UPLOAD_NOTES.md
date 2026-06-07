# Upload notes for paper-1109 (PET-DINO)

This `paper-1109/` directory was prepared for cloud upload from
`AutoSota-13/optimized_code/paper-1109/`.

## What was removed before upload
The following large directories were moved into `.removed_for_upload/`
(NOT included in the upload bundle):
- `data/`        (~820 MB) — COCO-style detection data
- `pretrained/`  (~2.2 GB) — pre-trained backbone / detector checkpoints

Both can be re-fetched from the original repository
(https://github.com/<paper-1109 PET-DINO repo>) or the paper's release page.

## Optimization summary
This paper's optimization was inference-only (DETR family — bipartite matching,
no NMS lever). The single applied change is:

    score_thr = 0.0  (instead of the default 0.05)

which yields AP 0.639 → 0.640 (+0.001).

The repository code itself is at the baseline `Update README.md` commit
(`7830a46`); the change above is a runtime parameter, not a source-code edit.
See `final_report.md` for the full trajectory.
