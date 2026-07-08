# Experimental Replication Guide

This document provides all commands needed to replicate the experiments from the paper. All commands are ready to copy-paste and execute.

## Prerequisites

- CUDA-enabled GPU (optional but recommended)
- Conda package manager
- Python 3.13

## Setup

### 1. Environment Setup

Create and configure the conda environment (also documented in README):

```bash
cd kmeans-trf
conda create -n kmt
conda activate kmt
conda install python=3.13 pip>25.0
conda install cudatoolkit -c anaconda  # Optional: only if GPU is available
pip install -r requirements.txt
```

### 2. Create Required Directories

```bash
mkdir -p runs results
```

---

## Training Runs

All training runs use the `expt-run.sh` script with the following parameters:
- `N`: Number of points
- `D`: Dimensionality
- `K`: Number of clusters
- `LR`: Learning rate
- `LI`: Loss inverse temperature ($\lambda^{-1}$)
- `T`: Training steps
- `TRIAL`: Trial number (1-10, corresponding to the 10 random seeds in the [`random_seeds`](random_seeds) file)
- `ENVNAME`: Conda environment name

**Note:** The 10 trials use the 10 random seeds provided in the [`random_seeds`](random_seeds) file:
```
153476998, 235510766, 205084180, 129959864, 149896109,
140439608, 145017219, 162518970, 245293503, 152674004
```

### Figure 9 - Loss Temperature Experiments

Train models with varying loss inverse-temperatures ($\lambda^{-1}$ = 1.0, 4.0, 7.0, 10.0) across dimensions 4 and 32:

```bash
for TRIAL in {1..10}; do
  for D in 4 32; do
    for LI in 1.0 4.0 7.0 10.0; do
      ENVNAME="kmt"; N=512; K=6; LR=0.01; T=10000; bash expt-run.sh ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
    done
  done
done
```

### Figure 10a - Number of Clusters Experiments

Train models with varying numbers of clusters (K = 10, 16, 25):

```bash
for TRIAL in {1..10}; do
  for K in 10 16 25; do
    ENVNAME="kmt"; N=512; D=32; LI=10.0; LR=0.01; T=10000; bash expt-run.sh ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
  done
done
```

### Figure 10b - Dimensionality Experiments

Train models with varying dimensions (D = 8, 16):

```bash
for TRIAL in {1..10}; do
  for D in 8 16; do
    ENVNAME="kmt"; N=512; LI=10.0; K=6; LR=0.01; T=10000; bash expt-run.sh ${N} ${D} ${K} ${LR} ${LI} ${T} ${TRIAL} ${ENVNAME}
  done
done
```

---

## Main Paper Figures

### Figure 3 and Figure 8 - Numerical Validation

See the Jupyter notebook: [`numerical_validation.ipynb`](numerical_validation.ipynb)

### Figure 4 - Generalization

All Figure 4 plots assume training results are in the `runs` directory.

#### Figure 4a - With One-Hot Encoding, Random Initialization

```bash
python plot_eval_ood_gen.py \
  -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results \
  -V "e:L" \
  -M 2.7 \
  -P "fig4a" \
  --niters 20 \
  --opmode mstep \
  --nlegendcols 1 \
  --seed 'E' \
  --init_scheme random \
  --quantile 0 \
  --skip_rembed_false \
  --alt_plot
```

#### Figure 4b - Without One-Hot Encoding, Random Initialization

```bash
python plot_eval_ood_gen.py \
  -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=none_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results \
  -V "e:L" \
  -M 2.7 \
  -P "fig4b" \
  --niters 20 \
  --opmode mstep \
  --nlegendcols 1 \
  --seed 'E' \
  --init_scheme random \
  --quantile 0 \
  --skip_rembed_false \
  --alt_plot
```

#### Figure 4c - With One-Hot Encoding, $k$-means++ Initialization

```bash
python plot_eval_ood_gen.py \
  -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results \
  -V "e:L" \
  -M 2.7 \
  -P "fig4c" \
  --niters 20 \
  --opmode mstep \
  --nlegendcols 1 \
  --seed 'E' \
  --init_scheme kmeams++ \
  --quantile 0 \
  --skip_rembed_false \
  --alt_plot
```

#### Figure 4d - Length Generalization
```bash
python plot_eval_ood_gen.py \
  -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results \
  -V "e:L" \
  -M 2.7 \
  -P "fig4d" \
  --niters 20 \
  --opmode lengen \
  --nlegendcols 1 \
  --seed 'E' \
  --init_scheme random \
  --quantile 0 \
  --skip_rembed_false \
  --alt_plot
```

### Figure 5 - Weight Visualization

See the Jupyter notebook: [`viz_weights.ipynb`](viz_weights.ipynb)

### Figure 6 - Attention Map Visualization

See the Jupyter notebook: [`viz_attn_maps.ipynb`](viz_attn_maps.ipynb)

---

## Appendix Figures

### Figure 9 - Loss Temperature Analysis

#### Figure 9a - 4 Dimensions

```bash
python plot_train_id_gen.py \
  -I runs \
  -R "n=512_N=512_k=6_d=4_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=*_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O results \
  -S 'l' \
  -V "e:L" \
  --numeric \
  -T "Loss temp" \
  -M 1.5 \
  --logy \
  -C "fig9a" \
  --quantile 25 \
  --drop_step_zero \
  --inv_hp_val \
  --nlegendcols 1 \
  --xticks_steps 5000 \
  --alt_plot
```

#### Figure 9b - 32 Dimensions

```bash
python plot_train_id_gen.py \
  -I runs \
  -R "n=512_N=512_k=6_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=*_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O results \
  -S 'l' \
  -V "e:L" \
  --numeric \
  -T "Loss temp" \
  -M 1.5 \
  --logy \
  -C "fig9b" \
  --quantile 25 \
  --drop_step_zero \
  --inv_hp_val \
  --nlegendcols 1 \
  --xticks_steps 5000 \
  --alt_plot
```

### Figure 10 - Scaling Analysis

#### Figure 10a - Varying Number of Clusters

```bash
python plot_train_id_gen.py \
  -I runs \
  -R "n=512_N=512_k=*_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O results \
  -S 'k' \
  -V "e:L" \
  --numeric \
  -T "nclusters" \
  -M 1.5 \
  --logy \
  -C "fig10a" \
  --quantile 25 \
  --drop_step_zero \
  --hp_val_int \
  --nlegendcols 1 \
  --xticks_steps 5000 \
  --alt_plot
```

#### Figure 10b - Varying Dimensions

```bash
python plot_train_id_gen.py \
  -I runs \
  -R "n=512_N=512_k=6_d=*_D=normal_s=0.1_em=True_es=True_sdb=True_e=*_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10.csv" \
  -O results \
  -S 'd' \
  -V "e:L" \
  --numeric \
  -T "ndims" \
  -M 1.5 \
  --logy \
  -C "fig10b" \
  --quantile 25 \
  --drop_step_zero \
  --hp_val_int \
  --nlegendcols 1 \
  --xticks_steps 5000 \
  --alt_plot
```

### Table 1 and Figure 11 - Varying Distribution Families

This command generates both Figure 11 and the data for Table 1:

```bash
python plot_eval_ood_gen.py \
  -I runs \
  -R "n=512_N=512_k=10_d=32_D=normal_s=0.1_em=True_es=True_sdb=True_e=onehot_q=1_A=softmax_a=1.0_p=0.01_L=softmax_g=False_l=10.0_E=*_b=32_r=0.01_C=0.5_P=5.0_T=10000_t=50_B=10_last.pt" \
  -O results \
  -V "e:L" \
  -M 2.7 \
  -P "tab1" \
  --niters 20 \
  --opmode varfam \
  --nlegendcols 1 \
  --seed 'E' \
  --init_scheme random \
  --quantile 0 \
  --skip_rembed_false \
  --alt_plot
```

---

## Notes

- All commands assume you are in the repository root directory (`kmeans-trf`)
- The conda environment `kmt` should be activated before running any commands
- Training runs will save results to the `runs/` directory
- Plotting commands will save figures to the `results/` directory
- Adjust the `ENVNAME` variable if you used a different conda environment name
