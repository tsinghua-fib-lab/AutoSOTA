# Dataset Card for InverseBench

- [Dataset Card for InverseBench](#dataset-card-for-inversebench)
  - [Data for linear inverse scattering](#data-for-linear-inverse-scattering)
  - [Data for MRI](#data-for-mri)
  - [Data for black hole imaging](#data-for-black-hole-imaging)
  - [Data for full waveform inversion](#data-for-full-waveform-inversion)
  - [Data for Navier-Stokes equation](#data-for-navier-stokes-equation)
  - [Acknowledgement](#acknowledgement)

Our data can be downloaded from [CaltechData](https://data.caltech.edu/records/zg89b-mpv16).

## Data for linear inverse scattering
- Generated using the online simulator [CytoPacq](https://cbia.fi.muni.cz/simulator/index.php)[1]. 
- License: Creative Commons BY-NC-SA 4.0
- Configuration used to generate data: 
  - VOI: `42x42x12`
  - Cover the whole CCD data: unchecked. 
  - Subpixel precision 1x. 
  - Type of phantom: HL60 nucleus (static)
  - Position: random
  - Amount: uniformly random number from 1 to 6.
  - Optical System: default value
  - Acquisition Device: default value
- Data post-processing: 
  - We crop each image to shape `128x128`.
  - Samples in test and val are selected so that their cosine similarity is less than 0.6 w.r.t the most similar sample in training set.


| Data Split | Number of Entries | Value range (min,max) | Unit |
|------------|-------------------|----------------------|------|
| Train      | 10,000            | (0, 1.0)            | F/m  |
| Test       | 100               | (0, 1.0)           | F/m  |
| Validation | 10                | (0, 1.0)            | F/m  |

We use three samples from the validation set to tune hyperparameters for each algorithm and 100 test samples for reporting the results.

## Data for MRI
- The multi-coil raw $k$-space data from the fastMRI knee dataset [2]. 
- We exclude the first and last 5 slices of each volume for training and validation as they do not contain much anatomical information and resize all images down to $320\times 320$ following the preprocessing procedure of the prior work [3]. 

| Data split | Number of entries | 
| ---------- | ----------------- | 
| Train      | 25,012            | 
| Test       | 94                | 
| Validation | 6                 | 

We use six samples from the validation set to tune hyperparameters for each algorithm and 94 test samples for reporting the results.

## Data for black hole imaging

- Training dataset is from GRMHD (50k).
- Synthetic test dataset from a different pretrained diffusion model.

| Data split | Number of entries | Value range (min, max) |
| ---------- | ----------------- | -----------------------|
| Train      | 50,000            | (0, 1)                 |
| Test       | 100               | (0, 1)                 |
| Validation | 5                 | (0, 1)                 |

We use five samples from the validation set to tune hyperparameters for each algorithm and 100 test samples for reporting the results.

## Data for full waveform inversion
- Adapted from the velocity map part of CurveFault dataset in OpenFWI by Deng, Chengyuan, et al. 2022 [1]. 
- License: Creative Commons BY-NC-SA 4.0
- Data adaptation by us: we resize the original velocity map from resolution 70x70 to 128x128 with bilinear interpolation and anti-aliasing. 

| Data split | Number of entries | Statistics (min,max) | Unit | 
|------------|-------------------|----------------------|------| 
| Train      | 50,000            | 1.50/4.50            | km/s | 
| Test       | 100               | 1.50/4.50            | km/s | 
| Validation | 10                | 1.50/4.50            | km/s |

We use one validation sample from the validation set to tune hyperparameters for each algorithm and 10 test samples for reporting the results due to the computational cost of the simulation.


## Data for Navier-Stokes equation
- We create a dataset of non-trivial initial vorticity fields by first sampling from a Gaussian random field and then evolving the 2D Navier-Stokes equation for five time units.
- The equation setup follows [5,6]. We set the Reynolds number to 200 and spatial resolution to 128$\times$128.

| Data split | Number of entries | Statistics (min,max) |
|------------|-------------------|----------------------|
| Train      | 20,480            | (-10, 10)            |
| Test       | 100               | (-10, 10)            |
| Validation | 10                | (-10, 10)            |

We use one validation sample from the validation set to tune hyperparameters for each algorithm and 10 test samples for reporting the results due to the computational cost of the simulation.


## Acknowledgement
- We thank Ben Prather, Abhishek Joshi, Vedant Dhruv, C.K. Chan, and Charles Gammie for
the synthetic blackhole images [GRMHD Dataset](https://iopscience.iop.org/article/10.3847/1538-4365/ac582e) used here, generated under NSF grant AST 20-34306. 


[1]: Wiesner D, Svoboda D, Maška M, Kozubek M. CytoPacq: A web-interface for simulating multi-dimensional cell imaging. Bioinformatics, Oxford University Press, 2019. ISSN 1367-4803. 2019. doi:10.1093/bioinformatics/btz417.

[2]: Zbontar, Jure, et al. "fastMRI: An open dataset and benchmarks for accelerated MRI." arXiv preprint arXiv:1811.08839 (2018).

[3]: Jalal, Ajil, et al. "Robust compressed sensing mri with deep generative priors." Advances in Neural Information Processing Systems 34 (2021): 14938-14954.

[4]: Deng, Chengyuan, et al. "OpenFWI: Large-scale multi-structural benchmark datasets for full waveform inversion." Advances in Neural Information Processing Systems 35 (2022): 6007-6020.

[5]: Iglesias, Marco A., Kody JH Law, and Andrew M. Stuart. "Ensemble Kalman methods for inverse problems." Inverse Problems 29.4 (2013): 045001.

[6]: Li, Zongyi, et al. "Physics-informed neural operator for learning partial differential equations." ACM/JMS Journal of Data Science 1.3 (2024): 1-27.
