# Installation Guide

## Install from PyPI
APPL is available on PyPI and can be installed with `pip`:
```bash
pip install -U applang
```

## Install from Source
To install APPL, clone the repository then install the `pip` package:
```bash
git clone https://github.com/appl-team/appl.git
cd appl
pip install -e .
```

Alternatively, if you do not need to modify the source code, you can install APPL from git directly with `pip`:

```bash
pip install git+https://github.com/appl-team/appl.git
```

## Check Installation
After running the above, you may verify your installation by running:
```bash
python -c "import appl"
```

## Install SGLang (Optional)
To use APPL with [SGLang](https://github.com/sgl-project/sglang) backend that serve local LLMs:
```bash
pip install "sglang[all]"
```
