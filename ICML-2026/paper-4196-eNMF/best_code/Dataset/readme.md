

# Dataset README

This folder contains all datasets (real and synthetic) used to reproduce the results in the paper.

---
## 1. Reproducing the results in the paper

1) **Get the datasets upon request** and place them into this repository under the `Dataset/` folder (preserve the `synthetic/` subfolder structure where noted).  

2) **Place each file at the specified location**

- **Face dataset**  
  - File: `face_id_4.npy` 
  - Destination: `Dataset/`

- **Verb dataset**  
  - File: `right_matrix.npy`  
  - Destination: `Dataset/verb/`

- **MovieLens dataset**  
  - File: `movielens1m_0.8training.npy`  
  - Destination: `Dataset/syn_dataset/`
  - **How to obtain:** Generate it with `Dataset/preprocess_movielens.py` using the raw MovieLens-1M data downloaded from <https://grouplens.org/datasets/movielens/1m/>.

- **Synthetic datasets**  
  - *Exact factorization — separable case*  
    - File: `synthetic/exacts_m_n_r_sparsity.npy`  
    - Destination: `Dataset/exact_dataset/`
  - *Exact factorization — general case*  
    - File: `synthetic/general_m_n_r_sparsity.npy`  (too large for github)
    - Destination: `Dataset/syn_dataset/`
  - *Noisy synthetic data*  
    - File: `synthetic/syn_noise_SNR_X.npy`  (too large for github)
    - Destination: `Dataset/syn_dataset/`

- **AudioMNIST datasets**  
  1. Follow the instructions in <https://github.com/soerenab/AudioMNIST> to **download and preprocess** the raw AudioMNIST data.  
  2. Run `python Dataset/preprocess_audiomnist_dataset.py` to produce the final file:  
     - **Output:** `Dataset/audiomnist.npy.npz`

3) **Preprocessing & data statistics**  
See **`Dataset_preparation_and_check.ipynb`** for preprocessing steps and dataset statistics.

## 2. Generate similar synthetic data
- Refer to **`snr_data_generation.ipynb`** for scripts and parameter settings to generate noisy synthetic datasets (e.g., different SNR levels).

---

## 3. Generate data for exact factorization
- For both separable and general exact-factorization datasets, also refer to **`snr_data_generation.ipynb`**. 

