# Final Report: paper-4019

- Title: Supervise Less, See More: Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting
- Primary metric: `AJI` (higher)
- Records: 10
- Generated: 2026-07-11T08:41:00Z

## Best Result

- Iteration: 9
- Idea: PARAM-01 — OT rho=0.8 + tuned NMS (best config)
- Primary metric: 0.604
- Commit: `1f69f0aad24abc5dec37beb2128471d61c9a389f`
- Notes: Best config: rho=0.8 + NMS sigma=0.2, containment=0.95, h_weight=0.5, merge_threshold=3. AJI +0.009 vs baseline. All 5 metrics improved. Pipeline uses DINOv2 ViT-L + SAM2.1 with PN/MoNuSeg png input (1024x1024).
