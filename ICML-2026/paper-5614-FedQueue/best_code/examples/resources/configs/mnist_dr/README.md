# Running Experiments with CADRE (Customizable Assurance of Data Readiness) Modules

This directory contains **CADRE Modules** designed to identify and handle various data readiness issues in the **MNIST dataset**.

## CADRE Module Overview

Inside the `cadre_module/` directory, we provide five different custom CADRE Modules. Each is responsible for detecting and optionally remedying a specific data readiness issue such as:

- Noise handling
- Class imbalance
- Duplicate samples
- Outlier management
- Memory usage optimization
- (Others depending on your implementation)

To run a specific CADRE module, configure it in the server configuration YAML file.

## Example: Handling Class Imbalance

To handle **class imbalance (ci)**, update the `cadremodule_configs` in `server_fedavg_cadremodule.yaml` as shown:

```yaml
cadremodule_configs:
    cadremodule_path: ./resources/configs/mnist_dr/cadre_module/handle_ci.py
    cadremodule_name: CADREModuleCI # Optional, if not specified, the last class will be used
    remedy_action: false  # Set to true to balance classes
```

The `dcadremodule_name` is the name of the class defined in the `cadremodule_path` file. It is optional, and if not specified, the last Python class defined within the CADRE module file will be used.
Setting `remedy_action: true` will apply the fix (e.g., undersampling the majority class). The agent will then perform:

- Evaluation of the custom data readiness status
- Remedial action (if enabled)

## Example: Handling Duplicates
To detect or fix **duplicate entries**, modify the YAML config like so:

```yaml
cadremodule_configs:
    cadremodulepath: ./resources/configs/mnist_dr/cadre_module/handle_duplicates.py
    cadremodulename: CADREModuleDuplicates # Optional, if not specified, the last class will be used
    remedy_action: false  # Set to true to remove duplicates
```

Each CADRE module defines its own metrics, rules and remedy mechanism.

## Running Experiments with MPI
To launch an experiment with 2 clients and 1 server using MPI, run the following while inside the `examples` directory:

```bash
mpiexec -n 3 python ./mpi/run_mpi.py \
    --server_config ./resources/configs/mnist_dr/server_fedavg_cadremodule.yaml \
    --client_config ./resources/configs/mnist_dr/client_1_cadremodule.yaml
```

## Running Experiments with gRPC
To launch an experiment with 2 clients and 1 server using gRPC, run the following three commands inside the `examples` directory from three different terminals:

```bash
# Terminal 1 [Server]
python ./grpc/run_server.py --config ./resources/configs/mnist_dr/server_fedavg_cadremodule.yaml
```

```bash
# Terminal 2 [Client1]
python ./grpc/run_client.py --config ./resources/configs/mnist_dr/client_1_cadremodule.yaml
```

```bash
# Terminal 3 [Client2]
python ./grpc/run_client.py --config ./resources/configs/mnist_dr/client_2_cadremodule.yaml
```

## DR Reports
After running, DR evaluation reports are saved in the `output/` directory.

Each report includes:

- General DR metrics and visualizations as enabled under `data_readiness_configs` in `server_fedavg_cadremodule.yaml`
- Custom metrics specific to the selected CADRE module

## Simulate data issues
Fortunately and unfortunately, most data issues are not naturally presented in the MNIST dataset as it is a very well-curated dataset, therefore, to evaluate CADRE modules effectively, we intentionally inject issues (e.g., duplicates, noise) to create a "noisy" and "non-AI-ready" version of MNIST dataset at `./resources/dataset/mnist_dataset_dr.py`.

## Notes
- You can extend the `BaseAgent` from `appfl.misc.data_readiness` and create new custom CADRE modules to handle any data related issues easily
