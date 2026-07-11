(chemtrain-deploy_installation)=
# Installation

## Building Connector

The connector interfaces XLA and PJRT with MD applications such as LAMMPS,
which might use a different building system and MPI.

Compiling the connector requires clang with C++14 support as well as the
NVCC CUDA compiler.

The typical GPU build command is:

```bash
python build.py --enable_cuda --cuda_version 12.6.0 \
    --cuda_compute_capabilities sm_80,sm_86
```

`--cuda_compute_capabilities` controls which GPU microarchitectures the
connector is compiled for.
Each value maps to a specific NVIDIA architecture:

| Capability | Architecture | Example GPUs |
|---|---|---|
| `sm_80` | Ampere | A100 |
| `sm_86` | Ampere | RTX 30xx, A40 |
| `sm_89` | Ada Lovelace | RTX 40xx |
| `sm_90` | Hopper | H100 |

Compiling for multiple capabilities increases binary size but allows the
same build to run on different GPU generations.
If you target a single machine, passing only its capability produces a
smaller and slightly faster binary.

The PjRt plugin for CUDA-enabled GPUs can be built alongside the connector:

```bash
python build.py --enable_cuda --cuda_version 12.6.0 \
    --cuda_compute_capabilities sm_80,sm_86 \
    --build_gpu_pjrt_plugin
```

Alternatively, a prebuilt PjRt plugin can be fetched from JAX.
Therefore, a JAX version compatible to the installed CUDA version and
compatible to the XLA library must be installed.
Then, the plugin can be fetched via

```bash
python build.py --enable_cuda --cuda_version 12.6.0 \
    --cuda_compute_capabilities sm_80,sm_86 \
    --load_gpu_pjrt_plugin
```

## Building LAMMPS Plugin

In the connector directory create and cd into a build directory and compile
the plugin with the following commands:

```bash
mkdir build && cd build
cmake -D LAMMPS_HEADER_DIR=<path/to/lammps/src> ../lammps_plugin
cmake --build .
```

**Note:** When the plugin is changed, it must be recompiled via

```bash
cmake --build . --clean-first
```

## Building LAMMPS with Plugin Support

To build lammps with plugin support, run:

```bash
cmake -D PKG_PLUGIN=yes ../cmake
cmake --build . -j <number_of_cores>
```

## "Installing" LAMMPS and the plugin

To "install" LAMMPS and the plugin, we can create a script to set the
correct environment variables. The script should look like this:

__activate:__ 
```bash
#! /bin/bash

export PATH=<path/to/lammps/build>:$PATH
export LAMMPS_PLUGIN_PATH=<path/to/chemtrain-deploy/build>
export JCN_PJRT_PATH=<path/to/chemtrain-deploy/lib>
```

Calling the script with ``source ./activate`` will set all necessary variables
to discover the LAMMPS executable, the plugin, and the PJRT library.

## Docker Container

**Note**: Using chemtrain-deploy within a docker container requires the 
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Before compiling the connector, determine the compute capabilities of the
GPUs:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

This prints one line per GPU, e.g. `8.0` for an A100 (`sm_80`).
Pass the result as `sm_<major><minor>` values to the build argument:

```bash
docker build -t chemtrain-deploy \
    --build-arg CUDA_COMPUTE_CAPABILITIES=sm_80,sm_86 \
    -f Dockerfile .
```

Afterward, simulations can be run inside the container:

```bash
docker run --gpus all -it --rm -v /home/ga27pej/myjaxmd/examples/spice:/workspace chemtrain-deploy
```
