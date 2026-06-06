# Project Management

## Package and Dependency Management
We use [pdm](https://pdm-project.org/) to manage the package and the dependencies.

## Pre-commit Hooks
We use [pre-commit](https://pre-commit.com/) to enforce code quality and consistency. The pre-commit hooks are defined in the `.pre-commit-config.yaml` file.

The following pre-commit hooks are enabled in this project.

### Code Linting and Formatting
We use [Ruff](https://github.com/astral-sh/ruff) for code linting and formatting. The format is based on [Black](https://github.com/psf/black).

The docstrings in the source code are written in the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) format. `pydocstyle` rules for checking the docstrings are included in `ruff`.

```bash
ruff check src
ruff format src
```

### Code Type Checking
We use [mypy](http://mypy-lang.org/) for static type checking. 

```bash
mypy src
```
