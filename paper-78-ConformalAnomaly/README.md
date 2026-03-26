# Conformal Anomaly Detection in Event Sequences
The implementation of our ICML-2025 paper ["Conformal Anomaly Detection in Event Sequences"](https://openreview.net/forum?id=Cq7XU5tmP8).


## Installation
1. Install the dependencies
    ```
    conda env create -f environment.yml
    ``` 
2. Activate the conda environment
    ```
    conda activate anomaly_tpp
    ```
3. Install the package (this command must be run in the `CADES` folder)
    ```
    pip install -e .
    ```
4. Unzip the data
    ```
    unzip data.zip
    ```

## Reproducing the results from the paper
- `experiments/spp.py`: GOF tests for SPP under nine alternatives (Section 4.1 in the paper).
- `experiments/multivariate.py`: Detecting anomalies in synthetic data (Section 4.2).
- `experiments/real_world.py`: Detecting anomalies in real-world data (Section 4.3).
- `experiments/fpr_control.py`: FPR control (Section 4.4).


## Citation
If you find this code useful, please consider citing our paper. Thanks.

```
@inproceedings{zhang2025conformal,
  title={Conformal Anomaly Detection in Event Sequences},
  author={Zhang, Shuai and Zhou, Chuan and Liu, Yang and Zhang, Peng and Lin, Xixun and Pan, Shirui},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

## Acknowledgements and References
Parts of this code are based on and/or copied from https://github.com/shchur/tpp-anomaly-detection.