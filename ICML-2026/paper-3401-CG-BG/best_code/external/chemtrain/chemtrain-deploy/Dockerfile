# Use the TensorFlow build image as the base

FROM tensorflow/build:2.20-python3.11

ARG CUDA_COMPUTE_CAPABILITIES="8.6"
ARG CUDA_CUDNN_VERSION="8.9.6"
ARG CUDA_VERSION="12.3.1"

ENV CUDA_VERSION=$CUDA_VERSION
ENV CUDA_CUDNN_VERSION=$CUDA_CUDNN_VERSION
ENV CUDA_COMPUTE_CAPABILITIES=$CUDA_COMPUTE_CAPABILITIES


RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    bzip2 \
    cmake \
    file \
    gcc \
    git \
    gfortran \
    g++ \
    libfftw3-dev \
    libjpeg-dev \
    libpng-dev \
    libhdf5-dev \
    make \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-h5py \
    python3-yaml \
    python3-lxml \
    python3-six \
    python3-setuptools \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /chemsim

# Compile the connector
COPY ./connector ./connector
COPY ./lammps_plugin ./lammps_plugin
COPY ./third_party ./third_party
COPY ./.bazelversion ./.bazelversion
COPY ./build.py ./build.py
COPY ./jax.bazelrc ./jax.bazelrc
COPY ./WORKSPACE ./WORKSPACE

# Build OpenMPI

RUN echo "Installing Open MPI" && \
    export OMPI_DIR=/opt/ompi && \
    export OMPI_VERSION=4.1.5 && \
    export OMPI_URL="https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-$OMPI_VERSION.tar.bz2" && \
    mkdir -p /tmp/ompi && \
    mkdir -p /opt && \
    cd /tmp/ompi && wget -O openmpi-$OMPI_VERSION.tar.bz2 $OMPI_URL && tar -xjf openmpi-$OMPI_VERSION.tar.bz2 && \
    cd /tmp/ompi/openmpi-$OMPI_VERSION && ./configure --prefix=$OMPI_DIR && make -j$(nproc) install && \
    export PATH=$OMPI_DIR/bin:$PATH && \
    export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH


RUN python build.py --build_gpu_pjrt_plugin --enable_cuda --cuda_version $CUDA_VERSION --cuda_compute_capabilities $CUDA_COMPUTE_CAPABILITIES --cudnn_version $CUDA_CUDNN_VERSION
RUN python build.py


# Point to OMPI binaries, libraries, man pages
ENV OMPI_DIR=/opt/ompi
ENV PATH="$OMPI_DIR/bin:$PATH"
ENV LD_LIBRARY_PATH="$OMPI_DIR/lib:$LD_LIBRARY_PATH"

# Fetch LAMMPS and build the plugin with LAMMPS
RUN git clone https://github.com/lammps/lammps.git && \
    cd lammps && git checkout stable_29Aug2024 && \
    mkdir build && \
    cd build && \
    cmake -D PKG_PLUGIN=yes ../cmake && \
    cmake --build . -j$(nproc)

RUN mkdir build && \
    cd build && \
    cmake -D LAMMPS_HEADER_DIR=/chemsim/lammps/src ../lammps_plugin && \
    cmake --build . -j$(nproc)

WORKDIR /

# Set all environment variables

ENV PATH="/chemsim/lammps/build:$PATH"
ENV LAMMPS_PLUGIN_PATH="/chemsim/build/"
ENV JCN_PJRT_PATH="/chemsim/lib"
