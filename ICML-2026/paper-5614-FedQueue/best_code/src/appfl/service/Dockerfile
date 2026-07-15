# Base image
FROM python:3.10

# Install system dependencies (Open MPI)
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    && apt-get clean

# Set the working directory
WORKDIR /app

# COPY all docs from  ../../.. to /app
COPY . /app

# Install the dependencies
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e ".[examples]"
RUN pip install globus-compute-sdk==3.15.0 globus-compute-endpoint==3.15.0 globus-sdk==3.65.0

# Change directory to examples
WORKDIR /app/src/appfl/service

# Run the app
ENTRYPOINT ["python", "appflx_entry_point.py"]
