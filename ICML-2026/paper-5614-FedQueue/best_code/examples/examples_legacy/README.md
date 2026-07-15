# 📝 Examples

**[DEPRECATED] This `examples_legacy` folder contains examples of the deprecated version of APPFL and will be removed in the future. Please also check examples in the `examples` folder.**


This directory contains examples showing how to run federated learning experiments using various communication protocols, such as MPI, gRPC, and GlobusCompute.
Currently, we have the following examples:

- MNIST
- CIFAR10
- FEMNIST
- CelebA
- Federated load forecasting with personalization layers
- FLamby Datasets
- Coronahack

## ⬇️ Prepare the Datasets

- MNIST and CIFAR10: As these two datasets are part of `torchvision`, so you only need to install the package `pip install torchvision`, and then those datasets will be automatically downloaded for you when you first run the experiments scripts on those datasets.
- Coronahack is a dataset from Kaggle, you can follow the download and preprocess instructions [here](datasets/RawData/README.md) before running the corresponding experiment scripts.
- FEMNIST and CelebA are two datasets from FL benchmark datasets [LEAF](https://github.com/TalwalkarLab/leaf/tree/master), you can follow the download instructions [here](datasets/RawData/README.md) to manually download them before running the corresponding experiment scripts.
- Federated load forecasting with personalization layers uses power consumption data from the NREL dataset, which will be automatically downloaded for you when you first run the experiments script.
- FLamby is a benchmarking dataset from cross-silo federated learning, and you need to download the datasets manually by following the [FLamby instructions](https://github.com/owkin/FLamby).

## 🚀 How to Run

We use separate python files for examples on various datasets with different communication protocols. Taking the MNIST dataset as an example,

- `mnist_serial.py` contains example on the partitioned MNIST dataset using serial simulation for synchronous federated learning.
- `mnist_mpi_sync.py` contains example on the partitioned MNIST dataset using MPI for synchronous federated learning.
- `mnist_mpi_async.py` contains example on the partitioned MNIST dataset using MPUI for asynchronous federated learning.
- `mnist_mpi_privacy.py` contains example on the partitioned MNIST dataset using MPI for synchrnous privacy-preserving federated learning.
- `mnist_grpc.py` contains example on the partitioned MNIST dataset using gRPC for synchronous federated learning.
- `mnist_globus_compute` contains example on the partitioned MNIST dataset using [Globus Compute](https://funcx.readthedocs.io/en/latest/) for synchronous or **asynchronous** federated learing.

### MNIST, CIFAR10, Coronahack, CelebA

Below shows how to run the MNIST examples. CIFAR-10 is pretty similar. For Coronahack and FEMNIST, check the dataset directory for preprocessing the data and replace the file name with an appropriate example name. For the input parameters and the corresponding meanings, please refer to the code itself.

#### Serial Run

The following command shows how to *simulate* federated learning by running FL server and five FL clients serially.

```bash
python ./mnist_serial.py --num_clients 5 --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10
```

#### MPI Synchronous Communication

`-np 6` in the following command means that we are launching 6 MPI processes. If you do not explicitly specify `--num_clients`, then the number of clients is the number of processes minus one (one MPI process is used as an FL server), which is 5 in this case. You can also explicitly specify the number of clients, e.g., `--num_clients 10` will launch ten FL clients with two clients running serially on one MPI process (also, one MPI process is used as an FL server).

```bash
mpiexec -np 6 python ./mnist_mpi_sync.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10
```

#### MPI Asynchronous Communication

For asynchronous communication, the number of clients is exactly equal to the number of MPI processes minus one, so we are launching 5 clients running FL asynchronously for the following commands. We do not allow more than one client to run serially on one MPI process for asynchronous cases as it is not making sense.

```bash
mpiexec -np 6 python ./mnist_mpi_async.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 20 --server ServerFedCompass
```

#### MPI Synchronous Communication with Special Privacy-Preserving Algorithms

APPFL provides two special synchronous privacy-preserving algorithms, [IIADMM](https://arxiv.org/pdf/2202.03672.pdf) and [ICEADMM](https://arxiv.org/pdf/2110.15318.pdf). We provide examples to showcase how to use them in `mnist_mpi_privacy.py`

```bash
mpiexec -np 6 python ./mnist_mpi_privacy.py --partition class_noiid --num_epochs 10 --server IIADMM
```

#### gRPC Synchronous Communication

For gRPC communication, we launch multiple MPI processes as well for simulation purposes only. Note that our gPRC implementation itself does not require any MPI communication, so you can also run FL server and several FL clients separately. (Asynchronous communication using gRPC is to be added.)

```bash
mpiexec -np 6 python ./mnist_grpc.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10
```

#### Globus Compute Communication

[Globus Compute](https://funcx.readthedocs.io/en/latest/index.html) is a distributed function as a service platform. It is used to support federated learning among **real-world** distributed and **heterogeneous** computing facilities. The following command starts an FL server which interacts with Globus Compute Endpoints (FL clients) to perform FL experiments. ***Please see the detailed instructions about how to setup experiments using Globus Compute [here](globus_compute/README.md).***

```bash
python mnist_globus_compute.py --client_config path_to_client_config.yaml --server_config path_to_server_config.yaml
```

### FEMNIST

For FEMNIST, the only difference is that it has more clients.

#### MPI communication

```bash
mpiexec -n 204 python femnist_mpi.py --server ServerFedAvg --num_epochs 6 --client_lr 0.01
```

#### gRPC communication

```bash
mpiexec -n 204 python femnist_grpc.py
```

### Load Forecasting with Personal FL

Personalization layers allow certain layers of the model to remain local to each client. For example, the load forecasting model is based on a LSTM+fully connected architecture. Suppose we want to personalize the fully connected layer's weights in a MPI setting, then this can be achieved as follows.

#### MPI Synchronous Communication

```bash
mpiexec -n 43 python personalization_fedloadforecast.py --personalization_layers FCLayer1.weight,FCLayer1.bias,FCLayer2.weight,FCLayer2.bias,FCLayer3.weight,FCLayer3.bias,prelu1.weight,prelu2.weight --personalization_config_name MyPersonalization
```

#### Serial Run

```bash
python personalization_fedloadforecast.py --personalization_layers FCLayer1.weight,FCLayer1.bias,FCLayer2.weight,FCLayer2.bias,FCLayer3.weight,FCLayer3.bias,prelu1.weight,prelu2.weight --personalization_config_name MyPersonalization
```

### FLamby

FLamby is a cross-silo FL benchmark, and running experiments on it is similar to experiments on MNIST, CIFAR-10, etc. You only need to make sure that you have already installed flamby and downloaded the datasets.

#### MPI Synchronous Communication

```bash
mpiexec -np 7 python flamby_mpi_sync.py --num_epochs 5 --dataset TcgaBrca --num_local_steps 50 --server ServerFedAvg
```

#### MPI Asynchronous Communication

```bash
mpiexec -np 7 python flamby_mpi_async.py --num_epochs 30 --dataset TcgaBrca --num_local_steps 100 --server ServerFedAsynchronous --val_range 1
```

#### Serially Run

```bash
python flamby_serial.py --num_clients 6 --num_epochs 5 --dataset TcgaBrca --num_local_steps 50 --server ServerFedAvg
```

### Using Lossy Compression (Experimental)

To use lossy compression for given examples, you need to follow the instructions in [Lossy Compression](../src/appfl/compressor/README.md) to install the required packages and run the experiments.

Then add the following arguments to the command line:

#### MPI Synchronous Communication

```bash
mpiexec -np 6 python ./mnist_mpi_sync.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 10 --enable_compression
```

#### MPI Asynchronous Communication
```bash
mpiexec -np 6 python ./mnist_mpi_async.py --partition class_noiid --loss_fn losses/celoss.py --loss_fn_name CELoss --num_epochs 20 --server ServerFedCompass --enable_compression
```
