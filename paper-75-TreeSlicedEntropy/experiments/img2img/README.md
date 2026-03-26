# Unpaired Image-to-Image Translation

This repository provides the code for Unpaired Image-to-Image Translation experiments, focusing on comparisons between our proposed **Partial Tree-Sliced Wasserstein (PartialTSW)** and other state-of-the-art methods.

---

## Credits

This work builds upon the foundations laid by several outstanding research projects and their publicly available codebases. We extend our sincere gratitude to the authors for their contributions:

* **Light Unbalanced Optimal Transport (LUOT)**:
    * Paper: [Light Unbalanced Optimal Transport](https://neurips.cc/virtual/2024/poster/94382) (NeurIPS 2024)
    * Repository: [milenagazdieva/LightUnbalancedOptimalTransport](https://github.com/milenagazdieva/LightUnbalancedOptimalTransport)
* **Light Schrödinger Bridge (LightSB)**:
    * Paper: [Light Schrödinger Bridge](https://arxiv.org/abs/2310.01174) (ICLR 2024)
    * Repository: [ngushchin/LightSB](https://github.com/ngushchin/LightSB)
* **Adversarially Learned Asymmetric Encoder (ALAE)**:
    * Repository: [podgorskiy/ALAE](https://github.com/podgorskiy/ALAE) (for the ALAE model architecture and pretrained weights)

---

## 🧪 Experimental Setup

This section details the steps to reproduce the Unpaired Image-to-Image Translation experiments.

### Environment Setup

We recommend using `conda` for managing dependencies.

1.  **Create and activate the Conda environment for UOT-FM:**
    ```bash
    conda activate partial
    ```
    *Note: This environment is also used for running PartialTSW, SW, Db-TSW, and ULightOT experiments.*

### Data Preparation

1.  **Download the FFHQ Latent Space:**
    The experiments are performed on the latent representations of the FFHQ dataset obtained from a pretrained ALAE model.
    ```bash
    python uot_alae --download
    ```

### Running Experiments

#### Our Method (PartialTSW) and Baselines (SW, Db-TSW, ULightOT)

To execute the experiments for **Partial Tree-Sliced Wasserstein (PartialTSW)**, Sliced Wasserstein (SW), Distributional Sliced Wasserstein (Db-TSW), and Unbalanced Light Optimal Transport (ULightOT):

```bash
bash exp.sh
```

To run experiment for UOT-FM:
```bash
bash exp_uot.sh
```