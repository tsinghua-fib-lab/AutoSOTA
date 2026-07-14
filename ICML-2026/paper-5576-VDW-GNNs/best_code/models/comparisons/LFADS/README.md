# `lfads-torch`: A modular and extensible implementation of latent factor analysis via dynamical systems

[![arXiv](https://img.shields.io/badge/arXiv-2309.01230-b31b1b.svg)](https://arxiv.org/abs/2309.01230) [![Gitter](https://img.shields.io/badge/chat-on%20gitter-brightgreen)](https://app.gitter.im/#/room/#lfads:gitter.im)

**If you use lfads-torch in published work, please cite our preprint:**
>Sedler AR, Pandarinath C. lfads-torch: A modular and extensible implementation of latent factor analysis via dynamical systems. arXiv preprint arXiv:2309.01230. 2023 Sep 3.

Latent factor analysis via dynamical systems (LFADS) is a variational sequential autoencoder that achieves state-of-the-art performance in denoising high-dimensional neural spiking activity for downstream applications in science and engineering [1, 2, 3, 4]. Recently introduced variants have continued to demonstrate the applicability of the architecture to a wide variety of problems in neuroscience [5, 6, 7, 8]. Since the development of the original implementation of LFADS, new technologies have emerged that use dynamic computation graphs [9], minimize boilerplate code [10], compose model configuration files [11], and simplify large-scale training [12]. Building on these modern Python libraries, we introduce `lfads-torch` &mdash; a new open-source implementation of LFADS designed to be easier to understand, configure, and extend.

# Installation
To create an environment and install the dependencies of the project, run the following commands:
```
git clone https://github.com/arsedler9/lfads-torch.git
conda create --name lfads-torch python=3.9
conda activate lfads-torch
cd lfads-torch
pip install -e .
pre-commit install
```

# Basic Walkthrough
## DataModule Configuration
The first step in applying `lfads-torch` to your dataset is to prepare your preprocessed data files. Save your data as `n_samples x n_timesteps x n_channels` arrays in the HDF5 format using the following keys:
- `train_encod_data`: Data to be used as input when training the model.
- `train_recon_data`: Data to be used as a reconstruction target when training the model.
- `valid_encod_data`: Data to be used as input when validating the model.
- `valid_recon_data`: Data to be used as a reconstruction target when validating the model.

Note that for both training and validation data, `encod_data` may be the same as `recon_data`, but they can be different to allow prediction of held out neurons or time steps.

Create a new configuration file for your dataset (e.g. `configs/datamodule/my_datamodule.yaml`). For single-session runs, set `datafile_pattern` to the path to your data file. For multi-session runs, set `datafile_pattern` to a `glob`-style pattern that matches all of your data files.
```
_target_: lfads_torch.datamodules.BasicDataModule
datafile_pattern: <PATH-TO-HDF5-FILE>
batch_size: <YOUR-BATCH-SIZE>
```

We provide preprocessed example data files from the Neural Latents Benchmark in `datasets`. With [`nlb_tools`](https://github.com/neurallatents/nlb_tools) installed in your environment, you can additionally use the `NLBEvaluation` extension to monitor NLB metrics while training `lfads-torch` models.

## Model Configuration
Next, you'll need to create a model configuration file that defines the architecture of your LFADS model (e.g. `configs/model/my_model.yaml`). We provide several examples in `configs/model`.

While these config files provide an easy way to get up and running with LFADS relatively quickly, these default hyperparameters are unlikely to be the best ones for your dataset. We recommend sweeping over architecture and regularization hyperparameters in order to maximize performance.

### Single-Session
You can find an example configuration for a single-session model at `configs/model/nlb_mc_maze.yaml`. You'll need to update the following values with your dataset-specific parameters:
- `encod_data_dim`: The `n_channels` dimension of `encod_data` from your data file.
- `encod_seq_len`: The `n_timesteps` dimension of `encod_data` from your data file.
- `recon_seq_len`: The `n_timesteps` dimension of `recon_data` from your data file.
- `readout.modules.0.out_features`: The `n_channels` dimension of `recon_data` from your data file.

### Multi-Session
You can find an example configuration for a multi-session model at `configs/model/rouse_multisession_PCR.yaml`. Note that multi-session models perform best with a pre-computed, PCR-based initialization. You can find a tutorial on computing these initializations in `tutorials/multisession`.

The `MultisessionReadin` and `MultisessionReadout` layers infer the input and output dimensionality automatically from your data files. You'll need to set `encod_seq_len` and `recon_seq_len` as specified above, but `encod_data_dim` should be set to the output dimensionality of the readin layers.

## Training a Model
To train a single model on your dataset, start with the `scripts/run_single.py` script. Edit the path specified by `RUN_DIR` to your desired model directory and edit the `overrides` argument to `run_model` to the following:
```
overrides={
    "datamodule": "my_datamodule",
    "model": "my_model",
}
```
This will tell `lfads-torch` to use the custom datamodule and model configurations you just defined. Running this script in your `lfads-torch` environment should begin optimizing a single model on your GPU if it is available. Logs and checkpoints will be saved in the model directory, and model outputs will be saved in `lfads_output_sess{i}.h5` when training is complete. Feel free to inspect `configs/single.yaml` and `configs/callbacks` to experiment with different `Trainer` arguments, alternative loggers and callbacks, and more.

As a next step, try specifying a random search in `scripts/run_multi.py` or a population-based training run in `scripts/run_pbt.py` and running a large-scale sweep to identify high-performing hyperparameters for your dataset!

# Advanced Usage
`lfads-torch` is designed to be modular in order to easily adapt to a wide variety of use cases. At the core of this modularity is the use of composable configuration files via Hydra [11]. User-defined objects can be substituted anywhere in the framework by simply modifying a `_target_` key and its associated arguments. In this section, we draw attention to some of the more significant opportunities that this modularity provides.

## Modular Reconstruction
Reconstruction losses play a key role in enabling LFADS to model the data distribution across data modalities. In the original model, a Poisson reconstruction trains the model to infer firing rates underlying binned spike counts. In subsequent extensions of LFADS, Gamma and Zero-Inflated Gamma distributions have been used to model EMG and calcium imaging datasets [5, 6]. In `lfads-torch`, we provide implementations of Poisson, Gaussian, Gamma, and Zero-Inflated Gamma reconstruction costs and allow the user to easily define their own reconstructions. These can be easily selected using the `reconstruction` argument in the model configuration.

## Modular Augmentations
Data augmentation is a particularly effective tool for training LFADS models. In particular, augmentations that act on both the input data and the reconstruction cost gradients have endowed the model with resistance to identity overfitting (coordinated dropout [3, 4]) and the ability to infer firing rates with spatiotemporal superresolution (selective backpropagation through time [6, 7]). Other augmentations can be used to reduce the impact of correlated noise [5]. In `lfads-torch`, we provide a simple interface for applying data augmentations via the `AugmentationStack` class. At a high level, the user creates the object by passing in a list of transformations and specifying the order in which they should be applied to the data batch, the loss tensor, or both. Separate `AugmentationStack`s are applied automatically by the `LFADS` object during training and inference, making it much easier to experiment with new augmentation strategies.

## Modular Priors
The LFADS model computes KL penalties between posteriors and priors for both initial condition and inferred input distributions, which are added to the reconstruction cost in the variational ELBO. In the original implementation, priors were multivariate normal and autoregressive multivariate normal for the initial condition and inferred inputs, respectively. In `lfads-torch`, the user can easily switch between priors using the `ic_prior` and `co_prior` config arguments or create custom prior modules by implementing `make_posterior` and `forward` functions. This gives users the freedom to experiment with alternative priors that may be more appropriate for certain brain areas and tasks.

# Want to know more?
Join the [Gitter room](https://app.gitter.im/#/room/#lfads:gitter.im) to chat with the creators and follow issues, pull requests, and releases. If you have never used Gitter, you can sign in with Element via your GitHub credentials after clicking the link. You can also join via [Matrix](https://matrix.to/#/#lfads:gitter.im). 

# References
1. David Sussillo, Rafal Jozefowicz, LF Abbott, and Chethan Pandarinath. LFADS – Latent Factor Analysis via Dynamical Systems. arXiv preprint arXiv:1608.06315, 2016.
1. Chethan Pandarinath, Daniel J O’Shea, Jasmine Collins, Rafal Jozefowicz, Sergey D Stavisky, Jonathan C Kao, Eric M Trautmann, Matthew T Kaufman, Stephen I Ryu, Leigh R Hochberg, et al. Inferring single-trial neural population dynamics using sequential auto-encoders. Nature Methods, 15(10):805–815, 2018.
1. Mohammad Reza Keshtkaran and Chethan Pandarinath. Enabling hyperparameter optimization in sequential autoencoders for spiking neural data. Advances in Neural Information Processing Systems, 32, 2019.
1. Mohammad Reza Keshtkaran*, Andrew R Sedler*, Raeed H Chowdhury, Raghav Tandon, Diya Basrai, Sarah L Nguyen, Hansem Sohn, Mehrdad Jazayeri, Lee E Miller, and Chethan Pandarinath. A large-scale neural network training framework for generalized estimation of single-trial population dynamics. Nature Methods, 19(12), 2022.
1. Lahiru N Wimalasena, Jonas F Braun, Mohammad Reza Keshtkaran, David Hofmann, Juan  ́Alvaro Gallego, Cristiano Alessandro, Matthew C Tresch, Lee E Miller, and Chethan Pandarinath. Estimating muscle activation from emg using deep learning-based dynamical systems models. Journal of Neural Engineering, 19(3):036013, 2022.
1. Feng Zhu*, Andrew R Sedler*, Harrison A Grier, Nauman Ahad, Mark Davenport, Matthew Kaufman, Andrea Giovannucci, and Chethan Pandarinath. Deep inference of latent dynamics with spatio-temporal super-resolution using selective backpropagation through time. Advances in Neural Information Processing Systems, 34:2331–2345, 2021.
1. Feng Zhu, Harrison A Grier, Raghav Tandon, Changjia Cai, Anjali Agarwal, Andrea Giovannucci, Matthew T Kaufman, and Chethan Pandarinath. A deep learning framework for inference of single-trial neural population activity from calcium imaging with sub-frame temporal resolution. Nature Neuroscience, 19(12), 2022.
1. Brianna M Karpowicz, Yahia H Ali, Lahiru N Wimalasena, Andrew R Sedler, Mohammad Reza Keshtkaran, Kevin Bodkin, Xuan Ma, Lee E Miller, and Chethan Pandarinath. Stabilizing brain-computer interfaces through alignment of latent dynamics. bioRxiv, 2022.
1. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32, 2019.
1. William Falcon and The PyTorch Lightning team. PyTorch Lightning, 3 2019. URL `https://github.com/Lightning-AI/lightning`.
1. Omry Yadan. Hydra - A framework for elegantly configuring complex applications. Github, 2019. URL `https://github.com/facebookresearch/hydra`.
1. Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E Gonzalez, and Ion Stoica. Tune: A research platform for distributed model selection and training. arXiv preprint arXiv:1807.05118, 2018.
