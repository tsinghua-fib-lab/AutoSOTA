# Third-party baselines

The estimators under `baselines/` are included **for comparison only**. They are
re-implementations / adaptations of methods proposed by other authors and remain
the intellectual property of their respective authors, under their original
licenses. If you use any of them, please cite the corresponding paper.

| File / dir | Method | Reference |
|:-----------|:-------|:----------|
| `MINE.py` | MINE | Belghazi et al., *Mutual Information Neural Estimation*, ICML 2018 |
| `InfoNCE.py` | InfoNCE / CPC | van den Oord et al., *Representation Learning with Contrastive Predictive Coding*, 2018 |
| `SMILE.py` | SMILE | Song & Ermon, *Understanding the Limitations of Variational MI Estimators*, ICLR 2020 |
| `DoE.py` | Difference-of-Entropies | McAllester & Stratos, *Formal Limitations on the Measurement of MI*, AISTATS 2020 |
| `KSG.py` | KSG | Kraskov, Stögbauer & Grassberger, *Estimating Mutual Information*, PRE 2004 |
| `MINDE.py`, `minde/` | MINDE | Franzese et al., *MINDE*, 2024 (diffusion code based on `github.com/CW-Huang/sdeflow-light`) |
| `nde/` (MAF, NAF, MDN) | Normalizing-flow density estimators | Papamakarios et al. (MAF, 2017); Huang et al. (NAF, 2018) |
| `MIENF.py`, `FLE.py`, `MRE.py`, `VCE.py`, `FastMI.py` | Variational / flow / copula MI estimators | See each file's header |
| `InfoNet_V1/` | InfoNet (our prior work) | https://github.com/datou30/InfoNet |

Please consult the upstream repositories for the exact license terms of each method.
