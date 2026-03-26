# Modified K-means Algorithm with Local Optimality Guarantees (ICML 2025)
The code implementation for "Modified K-means Algorithm with Local Optimality Guarantees", which is accepted to ICML 2025.

**TLDR**:
We show that the K-means algorithm does not always converge to a locally optimal solution, and propose a modified algorithm with local optimality guarantees, with the same computational complexity as the original K-means algorithm.

**Lay Summary**:
The K-means algorithm is one of the most widely used algorithms for clustering datasets into homogeneous groups. It also seems to be largely accepted, from at least the 1980s, that the K-means algorithm converges to locally optimal solutions. In this work we first show, by counterexample, that this is not true in general. We then develop simple modifications to the K-means algorithm which guarantee that it converges to a locally optimal solution, while also keeping the same computational complexity as the original K-means algorithm. We performed extensive experiments on both synthetic and real-world datasets and confirmed that the K-means algorithm does not always converge to locally optimal solutions in practice, while also verifying that our algorithms generate improved locally optimal solutions with reduced clustering loss.

## How to Use

The experiments were primarily conducted using **C++**, and the corresponding code and results can be found in the [cpp](./cpp) folder.

For convenience, we also provide a **Python** implementation, which is available in the [py](./py) folder.  

## Cite
[Modified K-means Algorithm with Local Optimality Guarantees](https://proceedings.mlr.press/v267/li25bt.html):

```bibtex
@InProceedings{li2025modified,
  title = {Modified K-means Algorithm with Local Optimality Guarantees},
  author = {Li, Mingyi and Metel, Michael R. and Takeda, Akiko},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages = {35656--35684},
  year = {2025},
  volume = {267},
  publisher = {PMLR}
}
```
