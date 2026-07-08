# Setup Guide

To successfully run `uv sync` and set up the environment, follow these steps.

## 1. Install UV

UV is a fast Python package installer and resolver. Install it using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your shell or run:

```bash
source $HOME/.local/bin/env
```

> [!NOTE]
> UV will be installed to `~/.local/bin`. Make sure this directory is in your PATH.

## 2. Install System Dependencies

The following system packages are required for building dependencies like `box2d-py` and `scipy`:

```bash
sudo apt-get update
sudo apt-get install -y swig gfortran pkg-config libopenblas-dev
```

## 3. Configure Python Version

Ensure your `pyproject.toml` restricts the Python version to be compatible with all dependencies (specifically `torch`, `scipy`, and `GPy`).

Edit `pyproject.toml`:

```toml
[project]
# ...
requires-python = ">=3.12, <3.13"
# ...
```

## 4. Sync Dependencies

Run `uv sync` to install the Python dependencies:

```bash
uv sync
```
