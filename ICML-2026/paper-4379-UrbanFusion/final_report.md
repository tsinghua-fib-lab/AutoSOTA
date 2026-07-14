# Final Report: paper-4379

- Title: UrbanFusion: Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations
- Primary metric: `R^2` (higher)
- Records: 9
- Generated: 2026-07-13T23:25:14Z

## Best Result

- Iteration: 7
- Idea: CODE-3b — 100 Optuna trials for kernel_ridge
- Primary metric: 91.05
- Commit: `5a2569a3c1050aea8baf10c33e8559ff956be582`
- Notes: Increased Optuna trials from 50 to 100 for kernel_ridge (Nystroem+RBF+Ridge). Best R^2=91.05% on SVI+OSM+Coords (RBF kernel, n_components=1797, gamma=0.00137, alpha=0.0136), +2.70pp over baseline 88.35%. +0.23pp over 50-trial best (90.82%). The additional trials found a slightly better configuration shifting best modality from SVI+OSM+POI+Coords to SVI+OSM+Coords.
