# Installation

This repo is a pure-Python codebase **except** for COCO’s Python interface (`cocoex`),
which may require a platform-specific installation.

## Python environment

Tested with Python `>=3.10`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

If `python3 -m venv` fails on Debian/Ubuntu, install venv support and retry:

```bash
sudo apt-get install -y python3-venv
```

If your system uses a versioned Python package name (e.g., Python 3.12), you may need:

```bash
sudo apt-get install -y python3.12-venv
```

### Troubleshooting: “externally-managed-environment” (PEP 668)

Some Linux distributions prevent installing packages into the system Python with pip.
If you see an `externally-managed-environment` error, do one of the following:

- Use a virtual environment (recommended): run the commands in the section above, or
- Use a Conda environment, then install with pip inside that environment.

## COCO / `cocoex` (required for BBOB experiments)

The COCO BBOB experiments use COCO’s Python API:

- Module import: `import cocoex`
- Suites: `bbob-noisy` and `bbob-largescale`

If `cocoex` is not available, you can still reproduce **external tasks** (RL/HPO/LQR/MLP),
but the COCO/BBOB experiments will fail.

### What “installed” means

For this repository, COCO is considered installed if the following succeeds:

```bash
python3 -c "import cocoex; print('cocoex OK')"
```

### Option A (if available): install via pip

Some environments provide a prebuilt package:

```bash
pip install cocoex
```

If this works, you are done.

### Option B: install from the official COCO repository

If `pip install cocoex` is unavailable, install COCO from source (recommended approach):

1. Obtain the COCO source tree (the `numbbo/coco` project).
2. Build/install its Python interface (`cocoex`) following the upstream instructions for your platform.
3. Verify `import cocoex` works (see the check above).

## Troubleshooting

### “ImportError: No module named cocoex”

- Install COCO as described above, or
- Run only the external tasks:

```bash
python3 tools/reproduce_all.py --workers 4 --suite external
```
