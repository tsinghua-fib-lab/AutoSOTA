Contributing
============

Welcome! We appreciate your interest in contributing to ``appfl``. You can contribute to the package either by reporting issues, suggesting enhancements, or directly contributing to the code base by pull requests (PR). Please follow the guidelines below to ensure a smooth contribution process.

Issues
------

Report bugs using APPFL GitHub's `Issue Tracker <https://github.com/APPFL/APPFL/issues>`_. A useful bug report has detail, background, and sample code. For example, try to include:

- A quick summary and/or background.
- Steps to reproduce:

  - Be specific!
  - Give sample code if you can.
- What you expected would happen.
- What actually happens.
- Any additional information that could help us.

  - Why you think this might be happening.
  - Things you tried that didn't work.
  - Etc.

Local Development and Pull Requests
-----------------------------------

When you want to develop ``appfl`` locally to create a pull request, you can follow the steps below to set up your development environment and test your changes before submitting a PR.

1. Open an issue if necessary.

- For small bugs, feel free to open a pull request directly.

- For larger bugs or enhancements, please open an issue first. Having an associated issue makes it easier to track changes and discuss proposals before you get started.

2. Clone the repository: **Note:** If you are an external contributor, you need to fork the repository and clone your fork.

.. code-block:: bash

    git clone https://github.com/APPFL/APPFL.git # Or your forked repository
    cd APPFL
    pip install -e ".[dev,mpi,examples]"

3. Create a new branch for your changes from the ``main`` branch.

.. note::

    We suggest naming your branch as ``developer_name/issue_number`` (e.g., ``zilinghan/issue42``) if your pull request is addressing an open issue, or ``developer_name/feature_name-start_date`` (e.g., ``kibaek/mpi-2025-01-01``) if you are working on a new feature.

.. code-block:: bash

    git checkout -b my-branch-name

4. Make your changes.

- If you've added code that should be tested, add tests.
- If you've changed APIs, update the documentation.

5. Ensure the test suite passes and your code lints. We ``pre-commit`` to enforce code style and formatting. Here is how you can install it and run it: make sure your PR passes the pre-commit checks.

.. code-block:: bash

    pip install pre-commit
    pre-commit install
    pre-commit run --all-files

6. Open the pull request on GitHub.
