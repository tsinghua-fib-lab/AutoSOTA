# base image with cuda 12.1 and pytorch 2.2.0
FROM ptebic.azurecr.io/public/aifx/acpt/stable-ubuntu2004-cu121-py310-torch221:biweekly.202403.1.v1

# need wget and gawk
RUN apt-get update && apt-get install -y wget gawk

# miniconda
RUN mkdir -p /root/miniconda3 && \
    wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /root/miniconda3/miniconda.sh && \
    /root/miniconda3/bin/conda init bash && \
    /root/miniconda3/bin/conda init zsh

# git
RUN apt-get update && apt-get install -y git

# environment variables
ENV PATH=/root/miniconda3/bin:$PATH

# working directory
WORKDIR /gamba

# install everything needed for mamba
RUN conda install python=3.12.2 && \
pip install nvidia-cuda-nvcc-cu12 && \
  pip install numpy==1.26.4 && \ 
  pip install torch==2.2.0 && \
  pip install packaging==23.2 && \
  pip install causal-conv1d==1.3.0.post1 && \
  pip install mamba-ssm==2.1.0 && \
  pip install flash_attn==2.5.9.post1 && \
  conda install conda-forge::threadpoolctl && \
  pip install defaults mkl && \ 
  pip install sequence-models &&\
  pip install evodiff && \
  pip install wandb==0.16.5 && \
  pip install alembic==1.13.1 && \
  pip install aniso8601==9.0.1 && \
  pip install biotite==0.41.1 && \
  pip install blinker==1.8.2 && \
  pip install cachetools==5.3.3 && \
  pip install cloudpickle==3.0.0 && \
  pip install deprecated==1.2.14 && \
  pip install docker==7.1.0 && \
  pip install einops==0.8.0 && \
  pip install entrypoints==0.4 && \
  pip install evodiff==1.1.0 && \
  pip install fair-esm==2.0.0 && \
  pip install fasteners==0.19 && \
  pip install flask==3.0.3 && \
  pip install graphene==3.3 && \
  pip install graphql-core==3.2.3 && \
  pip install graphql-relay==3.2.0 && \
  pip install greenlet==3.0.3 && \
  pip install griddataformats==1.0.2 && \
  pip install gunicorn==22.0.0 && \
  pip install itsdangerous==2.2.0 && \
  pip install joblib==1.4.2 && \
  pip install lmdb==1.4.1 && \
  pip install mako==1.3.5 && \
  pip install markdown==3.6 && \
  pip install mda-xdrlib==0.2.0 && \
  pip install mdanalysis==2.7.0 && \
  pip install mlflow==2.14.1 && \
  pip install mmtf-python==1.1.3 && \
  pip install mrcfile==1.5.0 && \
  pip install msgpack==1.0.8 && \
  pip install ninja==1.11.1.1 && \
  pip install nvidia-cublas-cu12==12.1.3.1 && \
  pip install nvidia-cuda-cupti-cu12==12.1.105 && \
  pip install nvidia-cuda-nvrtc-cu12==12.1.105 && \
  pip install nvidia-cuda-runtime-cu12==12.1.105 && \
  pip install nvidia-cudnn-cu12==8.9.2.26 && \
  pip install nvidia-cufft-cu12==11.0.2.54 && \
  pip install nvidia-curand-cu12==10.3.2.106 && \
  pip install nvidia-cusolver-cu12==11.4.5.107 && \
  pip install nvidia-cusparse-cu12==12.1.0.106 && \
  pip install nvidia-nccl-cu12==2.19.3 && \
  pip install nvidia-nvjitlink-cu12==12.5.40 && \
  pip install nvidia-nvtx-cu12==12.1.105 && \
  pip install opentelemetry-api==1.25.0 && \
  pip install opentelemetry-sdk==1.25.0 && \
  pip install opentelemetry-semantic-conventions==0.46b0 && \
  pip install pdb-tools==2.5.0 && \
  pip install querystring-parser==1.2.4 && \
  pip install scikit-learn==1.5.0 && \
  pip install sqlalchemy==2.0.31 && \
  pip install sqlparse==0.5.0 && \
  pip install threadpoolctl==3.5.0 && \
  pip install triton==2.3.1 && \
  pip install werkzeug==3.0.3 && \
  pip install wrapt==1.16.0


# default command to run when the container starts
CMD ["bash"]