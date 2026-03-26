# Distributed Conformal Prediction via Message Passing

The repository provides implementations of quantile-based distributed conformal prediction (Q-DCP) and histogram-based distributed conformal prediction (H-DCP) of the [paper](https://arxiv.org/pdf/2501.14544) [1]. *This code is based on the open-source repository of [federated conformal prediction (FCP)](https://github.com/clu5/federated-conformal) [2].*

## Abstract

Post-hoc calibration of pre-trained models is critical for ensuring reliable inference, especially in safety-critical domains such as healthcare. Conformal Prediction (CP) offers a robust post-hoc calibration framework, providing distribution-free statistical coverage guarantees for prediction sets by leveraging held-out datasets. In this work, we address a decentralized setting where each device has limited calibration data and can communicate only with its neighbors over an arbitrary graph topology. We propose two message-passing-based approaches for achieving reliable inference via CP: quantile-based distributed conformal prediction (Q-DCP) and histogram-based distributed conformal prediction (H-DCP). Q-DCP employs distributed quantile regression enhanced with tailored smoothing and regularization terms to accelerate convergence, while H-DCP uses a consensus-based histogram estimation approach. Through extensive experiments, we investigate the trade-offs between hyperparameter tuning requirements, communication overhead, coverage guarantees, and prediction set sizes across different network topologies.

## Quick Start

1. Install the requirements

```bash
conda create --name DCP --file requirement.txt
```

2. Run `Q-DCP.ipynb` or `H-DCP.ipynb` in the `notebooks` folder to reproduce the main figures of the paper.

## Reference

[1] Wen, H., Xing, H., & Simeone, O. (2025). Distributed Conformal Prediction via Message Passing. *ICML*.

[2] Lu, C., Yu, Y., Karimireddy, S. P., Jordan, M., & Raskar, R. (2023). Federated conformal predictors for distributed uncertainty quantification. *ICML*.

## Citation

```
@article{wen2025distributed,
  title={Distributed Conformal Prediction via Message Passing},
  author={Wen, Haifeng and Xing, Hong and Simeone, Osvaldo},
  journal={arXiv preprint arXiv:2501.14544},
  year={2025}
}
```
