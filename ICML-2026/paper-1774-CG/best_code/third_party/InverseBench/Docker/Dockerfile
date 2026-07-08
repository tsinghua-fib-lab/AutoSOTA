FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER user

RUN mkdir -p $HOME/miniconda3
RUN cd $HOME/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O  $HOME/miniconda3/miniconda.sh && \
    bash  $HOME/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf $HOME/miniconda3/miniconda.sh
RUN /bin/bash -c "source $HOME/miniconda3/bin/activate && conda init bash && \
    conda create -y -n ibench python=3.11 && \
    conda activate ibench && \
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia && \
    pip install omegaconf wandb accelerate piq hydra-core lmdb && \
    pip install pydantic --upgrade && \
    pip install devito==4.8.11 'dask[complete]' && \
    pip install h5py sigpy && \
    pip install numpy==1.26.4 ehtim pytest && \
    conda install -c conda-forge pynfft"

# Add conda initialization to .bashrc to make conda available in subsequent RUN commands
RUN echo 'source $HOME/miniconda3/bin/activate' >> ~/.bashrc && \
    echo 'conda activate ibench' >> ~/.bashrc

ENV PATH="$PATH:/home/user/.local/bin"