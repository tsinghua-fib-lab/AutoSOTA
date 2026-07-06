# Upload Notes for Agent-Trading-Arena

This project should be uploaded to the existing `MTMQuantAI/Agent-Trading-Arena` repository as the `decoupledmarket/` directory.

## Recommended Repository Layout

```text
Agent-Trading-Arena/
|-- Agent-Trading-Arena/   # Findings of EMNLP 2025 code
|-- decoupledmarket/       # DecoupledMarket code
|-- docs/
|-- images/
|-- requirement.txt
`-- README.md
```

Do not overwrite the original `Agent-Trading-Arena/` directory. It is the reproduction entry point for the Findings of EMNLP 2025 paper.

## Root README Add-On

Add a short paper navigator near the top of the root README:

```markdown
## Papers and Code

This repository hosts our agent-based trading simulation research line.

| Paper | Venue | Code |
| --- | --- | --- |
| Agent Trading Arena | Findings of EMNLP 2025 | `Agent-Trading-Arena/` |
| Evolving Quantitative Reasoning through Self-Play in Digital Twin Markets | ICML 2026 | `decoupledmarket/` |
```

Then keep the existing EMNLP 2025 introduction below it, or move it under a section named `Agent Trading Arena`.

## Upload Checklist

- Copy this folder as `decoupledmarket/` into the GitHub repository root.
- Keep `decoupledmarket/assets/figures/*.png`; these are required by the README.
- Keep `decoupledmarket/papers/decoupledmarket.pdf` if the paper PDF should be available locally.
- Do not upload generated runtime data from `save/`.
- Do not upload `.env`, `.pytest_cache/`, or `__pycache__/`.
- Tag the original code state for the EMNLP paper, for example `emnlp2025-release`.
- Tag the new DecoupledMarket state after upload, for example `decoupledmarket-v1`.
