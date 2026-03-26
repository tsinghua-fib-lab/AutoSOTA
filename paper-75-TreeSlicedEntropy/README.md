# Tree-Sliced Entropy Partial Transport

[![Conference](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/Conferences/2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📄 **Paper**: [Tree-Sliced Entropy Partial Transport](https://openreview.net/forum?id=41ZbysfW4h) (NeurIPS 2025)

> Tree-Sliced Entropy Partial Transport (PartialTSW) extends Tree-Sliced Wasserstein (TSW) distances to *unbalanced measures*. It has the *closed-form formulation* suitable for dynamic-support distributions such as those used in generative modeling. To our knowledge, no prior sliced-Wasserstein variant provides a closed-form formulation for unbalanced transport.

## Requirements
To install the required Python packages, run
```
conda env create --file=environment.yaml
conda activate partial
```

## Quick Start

```python
import torch
from tsw import PartialTSW, generate_trees_frames

# Initialize Partial Tree-Sliced Wasserstein Distance
tsw_obj = PartialTSW(
    ntrees=250,              # Number of trees
    nlines=4,                # Lines per tree
    p=2,                     # Norm order
    delta=2,                 # Temperature parameter for distance-based mass division
    mass_division='distance_based',  # Mass division method
    device='cuda'
)

# Generate sample data
N, M, d = 100, 100, 3
X = torch.randn(N, d, device='cuda')
Y = torch.randn(M, d, device='cuda')

# Generate tree frames
theta, intercept = generate_trees_frames(
    ntrees=250, 
    nlines=4, 
    d=d, 
    gen_mode="gaussian_orthogonal"
)

# Compute Partial Tree-Sliced Wasserstein Distance with unbalanced masses
# Use tensors for proper gradient flow and computation efficiency
total_mass_X = torch.tensor(0.8, device='cuda')
total_mass_Y = torch.tensor(0.6, device='cuda')

distance = tsw_obj(X, Y, theta, intercept, 
                   total_mass_X=total_mass_X, 
                   total_mass_Y=total_mass_Y)
print(f"Partial TSW Distance: {distance:.4f}")
```

## Experiments

The repository includes comprehensive experiments demonstrating the method's effectiveness across applications. Each experiment folder contains detailed instructions and implementation:

* **`experiments/point_cloud/`** - Point cloud gradient flow
* **`experiments/image_gen/`** - Image generation
* **`experiments/img2img/`** - Image-to-image translation

## Analysis

Additional analysis code is provided in the `analysis/` folder:

* **`runtime_plot/`** - Runtime comparisons between Partial Optimal Transport solvers and our methods
* **`convergence/`** - Code to confirm estimation stability

## Acknowledgments

Our codebase is based on work in Partial Optimal Transport and Tree-Sliced Wasserstein, including [Db-TSW](https://github.com/Fsoft-AIC/DbTSW) and [NonlinearTSW](https://github.com/thanhqt2002/NonlinearTSW).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{tran2025partialtsw,
    title={Tree-Sliced Entropy Partial Transport},
    author={Tran, Viet-Hoang and Tran, Thanh and Chu, Thanh and Le, Tam and Nguyen, Tan M.},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=41ZbysfW4h}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
