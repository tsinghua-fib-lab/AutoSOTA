# PGDSA: Physics-Guided Dynamic Sparse Attention Network for Gravitational Wave Detection

**Physics-Guided Dynamic Sparse Attention Network for Gravitational Wave Detection Across Ground and Space-Based Observatories**

> **Status:** Code release is in preparation. The complete implementation will be made publicly available upon acceptance of the manuscript.

## Overview

PGDSA is a dual-branch deep learning framework for joint gravitational wave (GW) signal detection and waveform reconstruction, designed to operate across both ground-based and space-based detector platforms. The framework integrates physics-informed feature extraction with data-driven representation learning through a gated cross-modal fusion mechanism.

### Architecture

```
Input Signal
    ├── Physics Branch (Time-Frequency)
    │     ├── Differentiable Wavelet Transform (32 learnable Morlet kernels)
    │     ├── Dual-Attention Enhancement Module
    │     └── Convolutional Projection
    │
    ├── Neural Branch (Data-Driven)
    │     ├── Improved WaveNet Encoder (8 layers, adaptive dilation, dynamic gating)
    │     └── Dynamic Sparse Transformer (block partitioning + Top-K sparsification)
    │
    └── Gated Cross-Modal Fusion → Detection Head + Extraction Head
```

### Key Features

- **Differentiable Wavelet Transform Layer**: Learnable Morlet wavelet kernels with frequency constraints aligned to detector sensitivity bands
- **Dynamic Sparse Attention**: Block-wise Top-K sparsification reducing attention complexity from O(T²d) to O(TLd), achieving ~6.7× speedup at sequence length 4096
- **Cross-Platform Capability**: Unified framework evaluated on ground-based (G2Net, GWOSC O3) and space-based (LISA) datasets with minimal configuration changes
- **Multi-Task Learning**: Joint optimization of detection and waveform reconstruction

## Datasets

| Dataset | Type | Platform | Task |
|---------|------|----------|------|
| [G2Net](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection) | Simulated | Ground-based (LIGO/Virgo) | Detection |
| GWOSC O3 | Real observations | Ground-based (LIGO) | Zero-shot detection |
| LISA Simulated | Simulated | Space-based | Detection + Reconstruction |

- G2Net data is available from the Kaggle competition page.
- LISA simulated data is generated following the protocol of [Zhao et al. (2023)](https://doi.org/10.1038/s42005-023-01334-6), with code available at [space_signal_detection_1](https://github.com/AI-HPC-Research-Team/space_signal_detection_1).

## Results Summary

### Ground-Based Detection (G2Net)

| Model | ROC-AUC | Parameters |
|-------|---------|------------|
| ResNet-50 | 0.880 | ~25.6M |
| Transformer | 0.880 | ~12.8M |
| CNN + Wavelet | 0.870 | ~8.5M |
| **PGDSA (Ours)** | **0.886** | **20.7M** |

### Space-Based Detection (LISA, TPR at FAR=1%)

PGDSA achieves >98% detection rate across EMRI, MBHB, BWD, and SGWB signals at SNR ≥ 30, with waveform reconstruction Overlap exceeding 0.95 for the majority of test samples.

## Requirements

The implementation is based on:

- Python 3.10
- PyTorch 2.7+
- NumPy, SciPy
- GWpy (for GWOSC data processing)

Detailed environment setup instructions will be provided upon code release.

## Repository Structure (Planned)

```
PGDSA-GW-Detection/
├── models/
│   ├── wavelet_branch.py        # Differentiable wavelet transform + enhancement
│   ├── wavenet_encoder.py       # Improved WaveNet with adaptive dilation
│   ├── sparse_transformer.py    # Dynamic sparse attention module
│   ├── fusion.py                # Gated cross-modal fusion
│   └── pgdsa.py                 # Complete PGDSA model
├── data/
│   ├── g2net_dataset.py         # G2Net data loading and preprocessing
│   ├── gwosc_dataset.py         # GWOSC O3 data processing
│   └── lisa_dataset.py          # LISA simulation data loading
├── configs/
│   ├── g2net_config.yaml        # Ground-based configuration
│   └── lisa_config.yaml         # Space-based configuration
├── scripts/
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference script
├── requirements.txt
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{pgdsa2026,
  title={Physics-Guided Dynamic Sparse Attention Network for Gravitational Wave Detection Across Ground and Space-Based Observatories},
  year={2026},
  note={Under review}
}
```

## License

This project will be released under the MIT License.

## Acknowledgments

We thank the LIGO Scientific Collaboration, Virgo Collaboration, and KAGRA Collaboration for making gravitational-wave data publicly available through the Gravitational Wave Open Science Center (GWOSC). We also thank the organizers of the G2Net Kaggle competition for providing the benchmark dataset.
