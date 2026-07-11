# Connector

The connector compiles the JAX model to HLO using python and provides an interface to
evaluate the model in C++ via a shared library.

## Docker Container

**Note**: Using chemtrain-deploy within a docker container requires the 
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Before compiling the connector, we have to determine the compute capabilities
of the GPUs. Therefore, we can run the following command

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

We then set the compute capabilities as environment variable and build the
docker container.

```bash
CUDA_COMPUTE_CAPABILITIES="8.6,8.0"

docker build -t chemtrain-deploy \
    --build-arg CUDA_COMPUTE_CAPABILITIES=${CUDA_COMPUTE_CAPABILITIES} \
    -f Dockerfile .
```

Afterward, simulations can be run inside the container:

```bash
docker run --gpus all -it --rm -v /home/ga27pej/myjaxmd/examples/spice:/workspace chemtrain-deploy
```

## Build Connector
The connector interfaces XLA and PJRT with MD applications such as LAMMPS,
which might use a different building system and MPI.

To build the connector, create an environment with python 3.11 and install JAX for GPU:
```bash
pip install "jax[cuda12]==0.4.37"
```

The connector can be built using the following command:

```bash
python build.py
```

Additionally, the PjRt plugin for CUDA enabled GPUs can be built using

```bash
python build.py --build_gpu_pjrt_plugin --enable_cuda --cuda_version 12.6.0
```

Alternatively, a prebuilt PjRt plugin can be fetched from JAX.
Therefore, a JAX version compatible to the installed CUDA version and 
compatible to the XLA library must be installed.
Then, the plugin can be fetched via

```bash
python build.py --load_gpu_pjrt_plugin
```


## Building LAMMPS Plugin

In the connector directory create and cd into a build directory and compile
the plugin with the following commands:

```bash
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CHEMTRAIN_DEPLOY_NATIVE_ARCH=ON \
      -D LAMMPS_HEADER_DIR=<path/to/lammps/src> ../lammps_plugin
cmake --build .
```

Release mode enables `-O3`; native-architecture optimization is enabled by
default and can be disabled for portable plugin binaries with
`-D CHEMTRAIN_DEPLOY_NATIVE_ARCH=OFF`.

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
