# Description of the codes

* datasets.py: Contains dataset loading and preprocessing logic.
* functional.py: Implements the dimensionality compression loss function.
* layers.py: Defines the convolutional blocks and overall network architecture used in the experiments.
* prepared_condigs.py: Stores commonly used hyperparameter settings for reproducibility.
* utils.py: Includes utility functions for building and managing the training pipeline.
* pipeline.py: Contains modular training components.
* training.py: Entry point for running all experiments described in the paper.
* analysis.py: Contains code for information breakdown analysis, an advanced method to theoretically estimate the mutual information carried by individual neurons and their synergistic combinations. This file supports the analysis in Figure 5c and related results in the paper.


# Instruction for reproductivity

The following provides a step-by-step instruction to reproduce ourwork.

1. Clone the repository to your local drive.
2. Create two directories, **./checkpoint/** (for saving trained model results) and **./data/** (for downloading the datasets).
3. Run the following command to call the script named `training.py` with the config file
   ```
   python training.py --config=./config.yaml
   ```
4. To run specific experiments, different experimental setups are wrapped as separate functions in training.py. You can manually call a specific function to execute the corresponding experiment. Please refer to the comments in the script for guidance.
5. To reproduce the theoretical information breakdown analysis (e.g., Fig. 4c), use `analysis.py`. 

# Citation

[Zhu, Zhichao, et al. "Stochastic Forward-Forward Learning through Representational Dimensionality Compression." arXiv preprint arXiv:2505.16649 (2025).](https://arxiv.org/abs/2505.16649)

   ```
@article{zhu2025stochastic,
  title={Stochastic Forward-Forward Learning through Representational Dimensionality Compression},
  author={Zhu, Zhichao and Qi, Yang and Ma, Hengyuan and Lu, Wenlian and Feng, Jianfeng},
  journal={arXiv preprint arXiv:2505.16649},
  year={2025}
}
   ```
