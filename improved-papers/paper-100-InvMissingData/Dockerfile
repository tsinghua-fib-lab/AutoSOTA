FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1
ENV CONDA_ENV_NAME=kpi


RUN conda create -n ${CONDA_ENV_NAME} python=3.10 -y

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    pip install  \
        uvicorn \
        fastapi \
        python-dotenv \
        requests \
        pandas \
        sqlalchemy \
        openpyxl \
        jupyterlab \
        matplotlib \
        seaborn \
        scikit-learn \
        tqdm \
        scipy \
        tensorboard \
        wandb \
        transformers \
        datasets"

RUN mkdir -p /workspace/data /workspace/logs /workspace/models

RUN touch /workspace/.initialized

WORKDIR /workspace
 
EXPOSE 8888 8000

ENTRYPOINT ["conda", "run", "-n", "kpi", "--"]

CMD ["bash"]
