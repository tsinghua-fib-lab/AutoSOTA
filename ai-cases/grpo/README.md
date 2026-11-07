# AutoSOTA · GRPO Research Handoff

### Protocol-Decoupled Advantage for shorter reasoning outputs

[Research report](REPORT.md) · [Reproduction](REPRODUCTION.md) · [Final measurements](results/matched500.csv) · [100-run ledger](results/exploration-ledger.csv)

AutoSOTA is an automated experimentation machine. This project records its investigation of GRPO-style post-training, including rejected ideas and the selected **PDA-Iso** mechanism.

| Selected method | Strongest accuracy control | Main observation |
|---|---|---|
| PDA-Iso | Matched DrGRPO | 56.91% fewer output tokens, with 9 fewer correct answers out of 1,024 |

PDA-Iso separates correctness and format credit, retaining full-response correctness updates while projecting format updates onto a fixed-mass protocol region. The final Countdown experiment uses Qwen2.5-0.5B, seed 0, 500 steps, and common adaptive Hint/SFT settings.

## Contents

- **REPORT.md** — detailed English research narrative, idea comparisons, failures, algorithm, final results, and future validation.
- **overlay/** — exact project changes at the selected historical commit, plus verifier sources and upstream license notices.
- **configs/** — recorded final runtime parameters for GRPO, DrGRPO, and PDA-Iso.
- **scripts/** — source-overlay installer and selected training launcher.
- **results/** — final measurements, intermediate checkpoints, exploration ledger, source hashes, and packaging checks.
- **assets/** — final accuracy–length figure in PNG.

Copy this folder into your project repository and retain its notices. This handoff does not push to any remote repository. Upstream framework installation, model weights, and Countdown data are external dependencies. Historical results were not rerun during packaging.

All project-specific idea proposal, experiment execution, and evaluation are attributed to AutoSOTA. Existing methods and upstream software retain their original attribution and licenses; see `overlay/LICENSE` and `overlay/Notice.txt`.
