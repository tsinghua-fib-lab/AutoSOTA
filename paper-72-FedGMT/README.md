# FL-SAM

Pytorch implementations of some federated learning methods based on sharpness-aware minimization.

I built this repository based on [PFLlib](https://github.com/TsingZ0/PFLlib) and [FL-Simulator](https://github.com/woodenchild95/FL-Simulator).  Thanks for their superior and understandable code architecture.

## Advantages

I did some optimization for time-saving and GPU-saving. For example, if you run FedAvg with 10% active clients per round of total 100 clients on CIFAR10 dataset with one NVIDIA 4090 GPU:

- Using two-layers CNN:  about only **2.72s runtime cost**  every round and **0.93GB GPU memory** cost.
- Using Resnet18:  about only **20s runtime cost**  every round and **1.86GB GPU memory** cost.



## Methods with code (updating)

- **FedAvg** — [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) *AISTATS 2017*

- **FedDyn** — [Federated Learning Based on Dynamic Regularization](https://openreview.net/forum?id=B7v4QMR6Z9w) *ICLR 2021*
- **FedSAM**—[Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf) ICML2022

- **FedSpeed**—[FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://openreview.net/pdf?id=bZjxxYURKT) ICLR 2023

- **FedSMOO**—[Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape](https://proceedings.mlr.press/v202/sun23h.html) ICML 2023
- **FedLESAM**—[Locally Estimated Global Perturbations are Better than Local Perturbations for Federated Sharpness-aware Minimization]([arxiv.org/pdf/2405.18890](https://arxiv.org/pdf/2405.18890)) ICML 2024
- **FedGMT**— [One Arrow, Two Hawks: Sharpness-aware Minimization for Federated Learning via Global Model Trajectory](https://openreview.net/pdf?id=80mK2Mqaph) ICML 2025



## Performance Evaluation

We show some results of  the CIFAR-10 dataset with 10% active clients per round of total 100 clients after 500 rounds. The corresponding hyperparameters are stated in the following. 

<p align="center">
<table>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="1">   </td>
            <td colspan="10"> CIFAR-10  </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="5">  CNN  </td>
            <td colspan="5">  ResNet18	 </td>
        </tr>
        <tr>
            <td colspan="1">  </td>
            <td colspan="1"> IID </td>
            <td colspan="1"> Dir-1.0 </td>
            <td colspan="1"> Dir-0.1 </td>
            <td colspan="1"> Dir-0.01 </td>
            <td colspan="1"> Time / round </td>
            <td colspan="1"> IID </td>
            <td colspan="1"> Dir-1.0 </td>
            <td colspan="1"> Dir-0.1 </td>
            <td colspan="1"> Dir-0.01 </td>
            <td colspan="1"> Time / round </td>
        </tr>
        <tr>
            <td colspan="1"> FedAvg </td>
            <td colspan="1"> 77.71 </td>
            <td colspan="1"> 75.96 </td>
            <td colspan="1"> 71.68 </td>
            <td colspan="1"> 63.27 </td>
            <td colspan="1"> 2.72s </td>
            <td colspan="1"> 76.74 </td>
            <td colspan="1"> 73.73 </td>
            <td colspan="1"> 64.34 </td>
            <td colspan="1"> 50.41 </td>
            <td colspan="1"> 20.10s </td>
        </tr>
        <tr>
            <td colspan="1"> FedDyn </td>
            <td colspan="1"> 77.94 </td>
            <td colspan="1"> 78.08 </td>
            <td colspan="1"> 76.71 </td>
            <td colspan="1"> 73.06 </td>
            <td colspan="1"> 3.01s </td>
            <td colspan="1"> 78.88 </td>
            <td colspan="1"> 77.89 </td>
            <td colspan="1"> 74.66 </td>
            <td colspan="1"> 69.41 </td>
            <td colspan="1"> 20.29s </td>
        </tr>
        <tr>
            <td colspan="1"> FedSAM </td>
            <td colspan="1"> 80.68 </td>
            <td colspan="1"> 78.33 </td>
            <td colspan="1"> 72.27 </td>
            <td colspan="1"> 63.80 </td>
            <td colspan="1"> 7.64s </td>
            <td colspan="1"> 77.71 </td>
            <td colspan="1"> 74.62 </td>
            <td colspan="1"> 64.38 </td>
            <td colspan="1"> 48.42 </td>
            <td colspan="1"> 40.38s </td>
        </tr>
        <tr>
            <td colspan="1"> FedSpeed </td>
            <td colspan="1"> 81.63 </td>
            <td colspan="1"> 81.66 </td>
            <td colspan="1"> 77.90 </td>
            <td colspan="1"> 74.24 </td>
            <td colspan="1"> 8.48s </td>
            <td colspan="1"> 79.58 </td>
            <td colspan="1"> 79.54 </td>
            <td colspan="1"> 75.66 </td>
            <td colspan="1"> 69.31 </td>
            <td colspan="1"> 42.11s </td>
        </tr>
        <tr>
            <td colspan="1"> FedSMOO </td>
            <td colspan="1"> 81.24 </td>
            <td colspan="1"> 80.98 </td>
            <td colspan="1"> 78.28 </td>
            <td colspan="1"> 75.28 </td>
            <td colspan="1"> 9.39s </td>
            <td colspan="1"> 79.50 </td>
            <td colspan="1"> 79.35 </td>
            <td colspan="1"> 75.22 </td>
            <td colspan="1"> 69.60 </td>
            <td colspan="1"> 43.07s </td>
        </tr>
        <tr>
            <td colspan="1"> FedLESAM-D </td>
            <td colspan="1"> 77.71 </td>
            <td colspan="1"> 78.69 </td>
            <td colspan="1"> 76.85 </td>
            <td colspan="1"> 72.71 </td>
            <td colspan="1"> 3.25s </td>
            <td colspan="1"> 78.45 </td>
            <td colspan="1"> 79.56 </td>
            <td colspan="1"> 74.82 </td>
            <td colspan="1"> 69.34 </td>
            <td colspan="1"> 22.32s </td>
        </tr>
        <tr>
            <td colspan="1"> FedGMT </td>
            <td colspan="1"> 81.62 </td>
            <td colspan="1"> 81.92 </td>
            <td colspan="1"> 79.45 </td>
            <td colspan="1"> 76.36 </td>
            <td colspan="1"> 3.21s </td>
            <td colspan="1"> 80.99 </td>
            <td colspan="1"> 80.10 </td>
            <td colspan="1"> 75.89 </td>
            <td colspan="1"> 70.28 </td>
            <td colspan="1"> 23.72s </td>
        </tr>
    </tbody>
</table>
</p>

**Common Training hyparameters**

In the above experiments, we employ SGD with a learning rate of 0.01, momentum of 0.9, weight decay of 1e-5, batch size of 50, local epoch of 5.

**Some key hyparameters selection**

<p align="center">
<table>
    <tbody align="center" valign="center">
        <tr>
            <td colspan="1">  </td>
            <td colspan="1"> SAM perturbation </td>
            <td colspan="1"> penalty coefficient </td>
            <td colspan="1"> others </td>
        </tr>
        <tr>
            <td colspan="1"> FedAvg </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedDyn </td>
            <td colspan="1"> - </td>
            <td colspan="1"> 10 </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedSAM </td>
            <td colspan="1"> {0.001,0.01,0.1} </td>
            <td colspan="1"> - </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedSpeed </td>
            <td colspan="1"> {0.001,0.01,0.1} </td>
            <td colspan="1"> 10 </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedSMOO </td>
            <td colspan="1"> {0.001,0.01,0.1} </td>
            <td colspan="1"> 10 </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedLESAM-D </td>
            <td colspan="1"> {0.001,0.01,0.1} </td>
            <td colspan="1"> 10 </td>
            <td colspan="1"> - </td>
        </tr>
        <tr>
            <td colspan="1"> FedGMT </td>
            <td colspan="1"> - </td>
            <td colspan="1"> 10 </td>
            <td colspan="1">  EMA coefficient α: {0.95, 0.995，0.998}<br>
                Sharpness strength γ: {0.5，1.0，2.0}</td>
        </tr>
    </tbody>
</table>
</p>



# Instructions

**Example codes to run FedGMT on CIFAR10 is given here.**

Please install the required packages. The code is compiled with Python 3.7 dependencies in a virtual environment via

```pip install -r requirements.txt```

## Code structure

- `./dataset`:
  - `utils`: code for heterogeneous partition strategy.
  - `generate_dataset.py`: generate client's local datasets .
- `./system`:
  - `main.py`: configurations of  methods. 
  - `./flcore`: 
    - `./clients/clientxxx.py`: the code on the client. 
    - `./servers/serverxxx.py`: the code on the server. 
    - `./trainmodel/models.py`: the code for backbones. 
  - `./utils`:
    - `mem_utils.py`: the code to record the GPU memory usage.
    - `data_utils.py`: the code to read the dataset. 
    - `xxx_utils.py`: the code for specific algorithm.

## 1. Generate Non-IID  Data:

### (1) Pathological non-iid data

if you want to generate Pathological non-iid data:

```bash
cd ./dataset 
python generate_dataset.py --shard -data cifar10 -nc 100 -shard_per_user 2 #Path(2)
```

### (2) Dirichlet non-iid data

if you want to generate Dirichlet non-iid data:

```bash
cd ./dataset 
python generate_dataset.py --LDA -data cifar10 -nc 100 -noniid 0.1 -if 0.5 #Dir(0.1) with long tail
```

## 2. Perform examples

### FedAvg

```bash
cd ./system
python main.py -algo FedAvg -data cifar10 -dev cuda --seed 1 -lr 0.01 -gr 500 -lbs 50 -le 5 -jr 0.1 -nc 100
```

### FedGMT

```bash
cd ./system
python main.py -algo FedGMT -data cifar10 -dev cuda --seed 1 -lr 0.01 -gr 500 -lbs 50 -le 5 -jr 0.1 -nc 100 -ga 1.0 -al 0.95 -tau 3.0 -be 10
```

## 3.Implement your own method

 To add a new algorithm, extend the base classes **Server** and **Client**, which are defined in `./system/flcore/servers/serverbase.py` and `./system/flcore/clients/clientbase.py`, respectively.



## Citation

If this codebase can help you, please cite our papers: 

```bibtex
@inproceedings{li2025one,
  title={One Arrow, Two Hawks: Sharpness-aware Minimization for Federated Learning via Global Model Trajectory},
  author={Li, Yuhang and Liu, Tong and Cui, Yangguang and Hu, Ming and Li, Xiaoqiang},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
