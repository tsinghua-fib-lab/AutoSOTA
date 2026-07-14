## Operating System & Hardware
We recommend using machines equipped with at least 8 cores, 16GB RAM, and ~10GB available disk space with **Ubuntu 20.04** or **Ubuntu 22.04** to stably reproduce the experimental results in our paper.  

## Software

- We develop BARO using Python 3.10.
- We encourage user to create a virtual environment to use BARO (e.g., `python3.10 -m venv env`).
- BARO could be installed using pip. (e.g., `pip install fse-baro`).
- BARO requires the following packages: "numpy", "pandas", "scikit-learn", "pytest", "tqdm", "requests", and "matplotlib". These dependencies are described in the [pyproject.toml](pyproject.toml) file.  We adhere to [PEP 735](https://peps.python.org/pep-0735/) for storing package requirements in the `pyproject.toml` file rather than the `requirements.txt` file.

