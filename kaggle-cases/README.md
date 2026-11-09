# Kaggle competition cases

Eight AutoSOTA competition case studies archived on November 9, 2026. Each case
includes source code and an English Markdown research report. Scores are from
the supplied historical submission records; they were not newly checked against
Kaggle during this repository update.

| Competition | Metric | Public | Private | Explore |
|---|---|---:|---:|---|
| **GeoHab 2026 MLWG** | Weighted F1 ↑ | 0.88296 | **0.88691** | [Report](geohab2026/REPORT.md) · [Code](geohab2026/) |
| **Novozymes Enzyme Stability Prediction** | Spearman ↑ | 0.60035 | **0.58370** | [Report](novozymes/REPORT.md) · [Code](novozymes/) |
| **Open Problems: Single-Cell Perturbations** | MRRMSE ↓ | 0.521 | **0.721** | [Report](c09/REPORT.md) · [Code](c09/) |
| **LEAP / ClimSim** | Weighted R² ↑ | 0.79431 | **0.79165** | [Report](leap/REPORT.md) · [Code](leap/) |
| **CHAMPS: Predicting Molecular Properties** | Mean per-type log MAE ↓ | −3.32551 | **−3.32019** | [Report](champs/REPORT.md) · [Code](champs/) |
| **G2Net Gravitational Wave Detection** | ROC-AUC ↑ | 0.88581 | **0.88540** | [Report](g2net/REPORT.md) · [Code](g2net/) |
| **Google Smartphone Decimeter Challenge 2021** | Position error (m) ↓ | 2.53616 | **1.12943** | [Report](gsdc2021/REPORT.md) · [Code](gsdc2021/) |
| **OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction** | MCRMSE ↓ | 0.22859 | **0.33917** | [Report](openvaccine/REPORT.md) · [Code](openvaccine/) |

Late-submission scores are comparisons with historical results, not changes to
official competition ranks. The reports distinguish reproduced controls, released
test-label feedback, public/private results, and incomplete upstream reproductions.
Metrics have different directions and scales and should not be pooled.

Only code, runtime configuration, dependency/license text, and Markdown reports
are included. Datasets, labels, weights, fitted arrays, predictions, submissions,
logs, and notebook outputs are excluded. See cleanup and integration notes (external reference: `INTEGRATION.md`).
