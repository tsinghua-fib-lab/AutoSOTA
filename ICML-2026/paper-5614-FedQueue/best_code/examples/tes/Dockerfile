# APPFL TES Client Docker Image
# This image provides the runtime environment for APPFL clients in TES containers

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for psutil
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy APPFL source code from the repository root
COPY . /appfl-source/

# Install APPFL from local source (includes most dependencies)
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu

RUN cd /appfl-source && pip install --no-cache-dir -e ".[examples]"

# Install additional TES-specific dependencies
RUN pip install --no-cache-dir requests scipy pyyaml pandas-plink

# Copy necessary resource files that clients need
RUN cp -r /appfl-source/examples/resources/ /app/resources/

# Create necessary directories
RUN mkdir -p /tmp /app/configs /app/data /app/output

# Symlink /tmp/resources -> /app/resources so that relative paths in configs
# (e.g. ./resources/...) resolve correctly when the TES executor sets workdir=/tmp
RUN ln -s /app/resources /tmp/resources

# Set environment variables
ENV PYTHONPATH=/app:/appfl-source/src
# Needed so that site_gwas_trainer.py can import gwas_config when APPFL writes
# the trainer to a temp path (breaking __file__-relative resolution).
ENV GWAS_PROJECT_DIR=/app/resources/configs/gwas

# Default command (will be overridden by TES task)
CMD ["python", "-c", "print('APPFL TES Client Container Ready')"]
