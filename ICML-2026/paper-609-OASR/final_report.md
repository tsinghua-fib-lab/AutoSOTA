# Final Report: paper-609

- Title: All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs
- Primary metric: `Accuracy` (higher)
- Records: 7
- Generated: 2026-07-05T07:37:57Z

## Best Result

- Iteration: 4
- Idea: CODE-05 — Cross-algorithm circuit selection: two_label_seed_43 (EP) + low_iou_0 (OASR)
- Primary metric: 100.0
- Commit: `46d18edb6131e7c9b80ab0fe94c485117d572578`
- Notes: Cross-algorithm circuit selection: Replaced low_iou_1 (OASR, ed=3.97%, edges=1289) with ep/two_label_seed_43 (Edge Pruning, ed=3.34%, edges=1085, acc=100%). Combined with low_iou_0 (ed=3.52%, edges=1145). Result: Edge Density 3.75->3.43% (-0.32pp, -8.5%), Edge Count 1217->1115 (-102, -8.4%). Accuracy maintained at 100%, Complement Acc at 46.5%. Both guardrails met. Circuit pair evaluated via standard eval_ioi_circuits.py protocol.
