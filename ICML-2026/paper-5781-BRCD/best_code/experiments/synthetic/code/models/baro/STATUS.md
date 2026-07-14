# STATUS 

We apply for the Available and Reusable badges.
We believe our artifact meets the requirements for the two badges in the [artifact submission guideline](https://2024.esec-fse.org/track/fse-2024-artifacts).

* **Available**: Our Software and Dataset artifacts are "Available" as they are publicly accessible through the following repository:
  * Software Artifacts: [https://github.com/phamquiluan/baro/releases/tag/0.2.0](https://github.com/phamquiluan/baro/releases/tag/0.2.0)
  * Dataset Artifacts: [https://zenodo.org/records/11046533](https://zenodo.org/records/11046533)


* **Reusable**: Our artifact is "Reusable" (and also "Functional") as we meet the following five criteria (the first four are the criteria for "Functional" badge) mentioned in the [artifact submission guideline](https://2024.esec-fse.org/track/fse-2024-artifacts).
  * _Documented_: We provide the following documents necessary for using our artifact: (1) [README.md](README.md), (2) [REQUIREMENTS.md](REQUIREMENTS.md), (3) [STATUS.md](STATUS.md), (4) [LICENSE](LICENSE), (5) [INSTALL.md](INSTALL.md), and (6) a copy of the accepted paper.
  * _Consistent & Complete_: We provide concrete steps for reproducing the main experimental results in the paper using our public artifacts, as described in Section [README.md#reproducibility](https://github.com/phamquiluan/baro/tree/main?tab=readme-ov-file#reproducibility).
  * _Exercisable_: We provide three tutorials at directory [./tutorials](tutorials) in the Jupyter Notebook format, which can also be opened using Google Colab. In addition, we also add unit tests for main functions at [./tests/test.py](./tests/test.py) file.
  * _Reusable for Future Researches_: We have published our BARO as a PyPI package [fse-baro](https://pypi.org/project/fse-baro/) and provided concrete instructions to install and use BARO (as described above). In addition, we also provide unit-tests and adopt Continuous Integration (CI) tools to perform testing automatically in daily manner to detect and prevent code rot, see [.circleci/config.yml](.circleci/config.yml). Thus, we believe that other researchers can use BARO in their own research.
