# Contributing

Thanks for helping make this artifact easier to reproduce.

## Development Setup

```bash
git clone <repo-url>
cd RTInfer
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

## Contribution Guidelines

- Keep simulator changes deterministic unless randomness is explicitly seeded.
- Do not commit generated experiment outputs, local datasets, model weights, or
  machine-specific paths.
- Prefer adding a small synthetic test for scheduler/layout changes.
- If a result depends on external Pantheon artifacts or Jetson hardware, mark it
  clearly as an integration or hardware reproduction step.
- Keep reviewer/rebuttal experiments separate from the original reproduction
  scripts unless a shared helper is clearly useful.

## Pull Request Checklist

- [ ] `python3 -m pytest` passes.
- [ ] New experiment scripts document their assumptions and output paths.
- [ ] Figures can be regenerated from committed code and documented inputs.
- [ ] No private paths, credentials, or tunnel addresses are committed.
