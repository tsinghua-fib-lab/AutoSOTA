# Documentation

This directory holds the Sphinx source for the OTP-FM documentation site,
published at <https://otp-fm.readthedocs.io>.

## Building locally

The pixi environment installs the docs toolchain (`sphinx`, `sphinx-rtd-theme`,
`myst-parser`, `myst-nb`, `sphinx-copybutton`) as part of the `otpfm[docs]`
editable install. From the repo root:

```bash
pixi install            # one-time
pixi run docs-html      # build into docs/_build/html
pixi run docs-clean     # remove docs/_build
open docs/_build/html/index.html
```

Without pixi:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Layout

```
docs/
├── conf.py                Sphinx configuration
├── index.md               Landing page + master toctree
├── installation.md
├── quickstart.md
├── customization.md
├── reproducibility.md     Includes ../REPRODUCIBILITY.md verbatim
├── tutorials/
│   ├── index.md
│   └── *.ipynb            Copied at build time from ../notebooks/
└── api/
    ├── index.md
    └── *.rst              automodule stubs for each otpfm.* module
```

The notebook copies under `docs/tutorials/*.ipynb` are gitignored - `conf.py`
mirrors `notebooks/*.ipynb` into this directory at build time so `myst-nb` can
pick them up.
