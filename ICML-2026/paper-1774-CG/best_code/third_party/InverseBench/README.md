
![](assets/logo.png)

[**InverseBench: Benchmarking Plug-and-Play Diffusion Priors for Inverse Problems in Physical Sciences**](https://openreview.net/pdf?id=U3PBITXNG6) (ICLR 2025 spotlight)

Hongkai Zheng*, Wenda Chu*, Bingliang Zhang*, Zihui Wu*, Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki, Zachary E. Ross, Katherine L. Bouman, Yisong Yue  
(*: Equal contribution)

Abstract: *Plug-and-play diffusion priors (PnPDP) have emerged as a promising research direction for solving inverse problems. 
 However, current studies primarily focus on natural image restoration, leaving the performance of these algorithms in scientific inverse problems largely unexplored. To address this gap, we introduce **InverseBench**, a framework that evaluates diffusion models across five distinct scientific inverse problems. These problems present unique structural challenges that differ from existing benchmarks, arising from critical scientific applications such as optical tomography, medical imaging, black hole imaging, seismology, and fluid dynamics. With **InverseBench**, we benchmark 14 inverse problem algorithms that use plug-and-play diffusion priors against strong, domain-specific baselines, offering valuable new insights into the strengths and weaknesses of existing algorithms.*
![](assets/teaser.png)

## Environment requirements
- We recommend Linux with 64-bit Python 3.11 for the best compatiblity.
- At least one high-end GPU for inference. All our experiments were conducted on A100 GPUs. 
- The main dependencies are `pytorch, scipy, hydra-core, lmdb, piq, wandb, accelerate, devito, dask, ehtim, ehtplot, h5py, sigpy, pynfft, pytest`. 
- Note: `pynfft` is required for the black hole imaging task and is only compatible with `numpy 1.xx`. If you encounter numpy compatibility issues, a simple workaround is to create a separate conda environment for black hole imaging.([issue #3](https://github.com/devzhk/InverseBench/issues/3))
- We also provide [Dockerfile](Docker/Dockerfile) under `Docker`, offering the best compatibility for all five tasks. It can be used as follows:
```bash
# Build docker image
docker build -t [image tag] --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

# Run docker container
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v [path to the top of this git repo]:/inverse-bench -v [path to data]:/data [image tag]
```
Breakdown of the `docker run` command:
- `--gpus all -it --rm`: With all GPUs enabled, run an interactive session, and automatically remove the container when it exits.
- `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`: Flags recommended by NVIDIA. Unlock the resource constraint.
- `-v [path to the top of this repo]:/inverse-bench -v [path to data]:/data`: Mount the current dir to `/inverse-bench`. Mount the data directory to `/data`.

## Pre-trained models

Pre-trained model weights can be found in the [GitHub release](https://github.com/devzhk/InverseBench/releases/tag/diffusion-prior) page.

| Problem  | Pretrained diffusion model| Training config |
|---------|---------------------------| ------------- |
| Full waveform inversion| `fwi-5m.pt` |  `configs/pretrain/fwi.yaml`|
| Linear inverse scattering| `in-scatter-5m.pt`| `configs/pretrain/inv-scatter.yaml`|
| 2D Navier-Stokes | `ns-5m.pt`| `configs/pretrain/navier-stokes.yaml` |
| Black hole | `blackhole.pt` | `configs/pretrain/blackhole.yaml`|
|MRI (brain) | `mri-brain.pt`| `configs/pretrain/brain-knee-mvue.yaml`|
|MRI (knee) | `mri-knee.pt`| `configs/pretrain/mri-knee-mvue.yaml`|
|FFHQ256 |  `ffhq256.pt` | `configs/pretrain/ffhq256.yaml`|

Note that `ffhq256.pt` is converted from the [DPS repository](https://github.com/DPS2022/diffusion-posterior-sampling). We used this model to verify that our reimplementation matches the results reported in the original paper.

You can also train your own diffusion prior via
```bash
accelerate launch --multi_gpu train.py -cn [specify config_name]
```
## Data
The test and validation data for each problem are available at the [data page](https://data.caltech.edu/records/jfdr4-6ws87).
A detailed description of each dataset can be found in [Dataset_Card.md](Dataset_Card.md).

## Inference

By default, `configs/config.yaml` will be loaded for inference. You can override the config value by
```bash
python3 main.py problem=[inverse problem config name] algorithm=[algorithm config name] pretrain=[pretrained model config name]
```
Note: The effective hyperparameter ranges for the same algorithm may vary significantly across different inverse problems. We recommend starting with the optimized hyperparameters provided in Table 12 of our paper as a reference point.

The structure of the inference config is explained below. 
| Key       | Description                                                                      |
|-----------|----------------------------------------------------------------------------------|
| `problem`   | Name of the inverse problem configuration. (See `configs/problem`)             |
| `algorithm` | Name of the algorithm configuration. (See `configs/algorithm`)                 |
| `pretrain`  | Name of the pre-trained model configuration. (see `configs/pretrain`)          |
| `tf32`      | (bool) Enables TF32 mode for improved performance on Ampere+ GPUs.             |
| `compile`   | (bool) Enable `torch.compile` (recommended for ensemble methods).              |
| `seed`      | (int) Random seed.                                                             |
| `inference` | (bool) If False, skip inference and only run evaluation.                       |
| `exp_name`  | (string) Sets the experiment name for logging and saving results.              |
| `wandb`     | (bool) Enables logging to Weights & Biases (WandB).                            |




## Hyperparameter search

To run hyperparameter sweeps using Wandb, follow these steps:

1. Run `wandb login` in the command line and follow the prompts to authenticate

2. Navigate to the `configs/sweep` directory, where you'll define sweep configuration files. These files define the hyperparameters to be tuned and their ranges for each specific problem and algorithm combination. See  `configs/sweep/inv-scatter/dps.yaml` for a concrete example.  
   

3. To start a sweep, use the following command:

   ```bash
   wandb sweep configs/sweep/[problem]/[algorithm].yaml
   ```

   For example, 

   ```bash
   wandb sweep configs/sweep/inv-scatter/dps.yaml
   ```

4. Wandb will generate a sweep ID in the form of `[your-entity]/[your-project]/[sweep-id]`. Use this ID to start sweep agents:

   ```bash
   wandb agent [your-entity]/[your-project]/[sweep-id] --count 10
   ```
   You can add count flag to set the maximum number of runs to try. 

5. You can start multiple agents to run the sweep in parallel by opening new terminals and running the same command `wandb agent [your-entity]/[your-project]/[sweep-id] --count 10`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
```bibtex
@inproceedings{
zheng2025inversebench,
title={InverseBench: Benchmarking Plug-and-Play Diffusion Priors for Inverse Problems in Physical Sciences},
author={Hongkai Zheng and Wenda Chu and Bingliang Zhang and Zihui Wu and Austin Wang and Berthy Feng and Caifeng Zou and Yu Sun and Nikola Borislavov Kovachki and Zachary E Ross and Katherine Bouman and Yisong Yue},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=U3PBITXNG6}
}
```


## Troubleshooting
This section contains the solutions to common issues. 

**Devito getting stuck?**
If your code gets stuck at `codepy/jit.py`, 
run the script [scripts/fwi/clear_devito_cache.py](https://github.com/devitocodes/devito/blob/master/scripts/clear_devito_cache.py) to make sure the cache is in a consistent state.  
