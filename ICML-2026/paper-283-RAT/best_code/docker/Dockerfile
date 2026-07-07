FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
ENV TZ=UTC \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential cmake qtbase5-dev qt5-qmake

RUN python3 -m pip install gym3
RUN python3 -m pip install https://github.com/openai/procgen/archive/refs/tags/0.10.7.zip
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# singularity compatibility: https://github.com/qgis/QGIS/issues/46841
RUN strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/

WORKDIR /workspace
