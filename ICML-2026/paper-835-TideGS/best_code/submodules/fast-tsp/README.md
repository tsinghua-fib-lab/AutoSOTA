# fast-tsp

[![Documentation Status](https://readthedocs.org/projects/fast-tsp/badge/?version=latest)](https://fast-tsp.readthedocs.io/en/latest/?badge=latest)
[![Pip Actions Status][actions-pip-badge]][actions-pip-link]
[![PyPI - Version](https://img.shields.io/pypi/v/fast-tsp)][pypi]
[![GitHub issues](https://img.shields.io/github/issues/shmulvad/fast-tsp?style=flat-square)](https://github.com/shmulvad/fast-tsp/issues)
[![GitHub license](https://img.shields.io/github/license/shmulvad/fast-tsp?style=flat-square)][LICENSE]


A library for computing near optimal solution to large instances of the TSP (Travelling Salesman Problem) fast using a local solver. The library is written in C++ and provides Python bindings.

# Quickstart

First install the library

```bash
$ pip install fast-tsp
```


Then run the problem on your 2D distance matrix:

```python
import fast_tsp
dists = [
    [ 0, 63, 72, 70],
    [63,  0, 57, 53],
    [72, 57,  0,  4],
    [70, 53,  4,  0],
]
tour = fast_tsp.find_tour(dists)
print(tour)  # [0, 1, 3, 2]
```

### Documentation

Documentation can be found at <https://fast-tsp.readthedocs.io/>.

You can build the documentation by `cd`ing to `docs/` and running `make clean && make html`.

### Citation

If you find that this project helps your research, please consider citing it using the metadata from the `CITATION.cff` file.

### License

This library is licensed under the MIT license. Additionally, it uses `pybind11` which is provided under a BSD-style license that can be found in the [LICENSE] file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

[LICENSE]: https://github.com/shmulvad/fast-tsp/blob/main/LICENSE
[actions-pip-link]: https://github.com/shmulvad/fast-tsp/actions?query=workflow%3APip
[actions-pip-badge]: https://github.com/shmulvad/fast-tsp/workflows/Pip/badge.svg
[pypi]: https://pypi.org/project/fast-tsp/
