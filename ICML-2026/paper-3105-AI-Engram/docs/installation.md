# Installation

## From PyPI

```bash
pip install ai-engram
```

This installs everything: `torch`, `tqdm`, and `transformers` — so HuggingFace
LLMs and the GPT-2 `Conv1D` path work out of the box. The distribution is named
**`ai-engram`**; the **import package is `engram`**:

```python
import engram
from engram import EngramEditor, EditorConfig
print(engram.__version__)
```

## With uv (notebook / Jupyter)

[uv](https://docs.astral.sh/uv/) is a fast installer. Create an isolated
environment and add `ai-engram` plus a Jupyter kernel — handy for running the
[example notebooks](https://github.com/jeakwon/ai-engram/tree/main/examples):

```bash
uv venv                         # create .venv (Python >= 3.9)
source .venv/bin/activate
uv pip install ipykernel ai-engram transformers torch
```

`ai-engram` already pulls `torch` and `transformers`; listing them explicitly
keeps the notebook environment fully reproducible. Register the kernel so it
shows up in Jupyter:

```bash
python -m ipykernel install --user --name ai-engram --display-name "Python (ai-engram)"
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- `transformers` (installed automatically)

## From source

```bash
git clone https://github.com/jeakwon/ai-engram
cd ai-engram
pip install -e ".[dev]"        # editable install + pytest
```

The package uses a `src/` layout (`src/engram`) with the
[hatchling](https://hatch.pypa.io) build backend.

## Verify

```bash
python -c "import engram; print(engram.__version__)"
pytest tests/test_extraction.py -q      # fast CPU unit tests
```

The heavy TOFU integration tests are gated behind environment variables and
require a GPU + cached models — see [TOFU validation](tofu.md).