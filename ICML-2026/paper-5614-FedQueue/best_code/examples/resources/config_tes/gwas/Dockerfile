# APPFL TES Client Docker Image — GWAS
# Multi-stage build: builder installs everything, final image contains only runtime artifacts.
# Build from the repository root:
#   docker build --no-cache -f examples/resources/config_tes/gwas/Dockerfile -t appfl/client-gwas:latest .

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Build-time system deps (gcc etc. needed to compile C extensions like psutil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (no torchvision/torchaudio — not used by GWAS)
RUN pip install --no-cache-dir \
    torch==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy only the package source needed for pip install
COPY src/ /appfl-source/src/
COPY setup.py pyproject.toml README.md /appfl-source/

# Install base APPFL (no [examples] extra — avoids opencv, jupyter, tensorboard, etc.)
RUN pip install --no-cache-dir -e /appfl-source

# Install GWAS/TES runtime dependencies
RUN pip install --no-cache-dir \
    requests \
    scipy \
    pyyaml \
    pandas-plink \
    matplotlib \
    scikit-learn

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the APPFL source (needed for the editable install's .pth / direct imports)
COPY --from=builder /appfl-source /appfl-source

# Copy only the resource files the GWAS clients need
COPY examples/resources/ /app/resources/

# Create necessary directories
RUN mkdir -p /app/configs /app/data /app/output

# Symlink /tmp/resources -> /app/resources so that relative paths in configs
# (e.g. ./resources/...) resolve correctly when the TES executor sets workdir=/tmp
RUN ln -s /app/resources /tmp/resources

# Set environment variables
ENV PYTHONPATH=/app:/appfl-source/src
# Needed so that site_gwas_trainer.py can import gwas_config when APPFL writes
# the trainer to a temp path (breaking __file__-relative resolution).
ENV GWAS_PROJECT_DIR=/app/resources/configs/gwas

# Default command (will be overridden by TES task)
CMD ["python", "-c", "print('APPFL TES GWAS Client Container Ready')"]
