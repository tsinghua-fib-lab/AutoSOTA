# Development
We use [pdm](https://pdm-project.org/latest/) to manage the dependencies and [MkDocs](https://www.mkdocs.org/getting-started/) to build the documentation.

## Pre Commit
It is recommended to use [pre-commit](https://pre-commit.com/) to ensure that the code is formatted and linted before committing.

### Steps
- Install pre-commit
    ```bash
    pip install pre-commit
    ```
- Install the git hooks
    ```bash
    pre-commit install
    ```
- Run the pre-commit checks (should automatically run when you try to commit, you can also run it manually)
    ```bash
    pre-commit run
    ```

## Coverage

To generate [coverage report](../coverage.md), run the following command:

```bash
coverage run -m pytest
coverage html
```
