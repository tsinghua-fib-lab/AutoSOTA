# =============================================================================
# vLLM Kernel Development Image
#
# Design:
# 1. Base: Inherit from official vLLM image, reuse PyTorch, Triton and uv.
# 2. Complete: Install nvcc and cuda-libraries-dev for .cu compilation.
# 3. Override: Uninstall pre-installed vLLM for local source mounting.
# 4. Accelerate: Use uv package manager and ccache for faster builds.
# =============================================================================
FROM vllm/vllm-openai:v0.13.0

USER root
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. Basic Config: APT Cache
# -----------------------------------------------------------------------------
# Configure APT to keep downloaded packages for faster rebuilds
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# -----------------------------------------------------------------------------
# 2. Install CUDA Dev Environment & System Tools
# -----------------------------------------------------------------------------
# Install compilers, CMake, Ninja and CUDA headers
# Add libnccl-dev for distributed communication
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git vim curl wget tmux \
    build-essential cmake ninja-build pkg-config \
    cuda-nvcc-12-9 \
    cuda-libraries-dev-12-9 \
    libnccl-dev \
    ccache \
    fonts-noto-cjk fonts-wqy-zenhei \
    # Profiling tools (optional)
    nsight-compute-2025.3.0 \
    nsight-systems-2025.3.2

# Enable ccache for incremental C++ compilation
ENV PATH="/usr/lib/ccache:${PATH}"
ENV CCACHE_DIR=/root/.ccache

# -----------------------------------------------------------------------------
# 3. Install Dependencies: cuSPARSELt (Custom Kernel Dependency)
# -----------------------------------------------------------------------------
# Auto-select cuSPARSELt 0.8.1 based on architecture
RUN export ARCH=$(dpkg --print-architecture) && \
    apt-get update && \
    apt-get remove -y libcusparselt0 libcusparselt-dev || true && \
    if [ "${ARCH}" = "amd64" ]; then \
        CUSPARSE_URL="https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-repo-ubuntu2204-0.8.1_0.8.1-1_amd64.deb"; \
    elif [ "${ARCH}" = "arm64" ]; then \
        CUSPARSE_URL="https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb"; \
    else \
        echo "Unsupported architecture: ${ARCH}" && exit 1; \
    fi && \
    wget -q ${CUSPARSE_URL} -O /tmp/cusparselt-repo.deb && \
    dpkg -i /tmp/cusparselt-repo.deb && \
    cp /var/cusparselt-local-repo-ubuntu2204-0.8.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends cusparselt-cuda-12 && \
    rm -f /tmp/cusparselt-repo.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "✅ cuSPARSELt 0.8.1 installed"

# -----------------------------------------------------------------------------
# 4. Cleanup: Uninstall Conflicting Packages
# -----------------------------------------------------------------------------
# Use uv (faster than pip) to uninstall pre-installed vLLM
# Critical for pip install -e . to work properly
RUN uv pip uninstall --system vllm

# Uninstall pip-installed cusparselt, use system version instead
RUN uv pip uninstall --system nvidia-cusparselt-cu12 || true

# -----------------------------------------------------------------------------
# 5. Build Configuration
# -----------------------------------------------------------------------------
# Limit target CUDA architectures to reduce JIT compile time
# 8.0=A100, 9.0=H100/H200, 10.0=B100/B200
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0"

# Control compilation parallelism:
# MAX_JOBS for Ninja parallel files, NVCC_THREADS for compiler threads
# Limit both to prevent OOM
ENV MAX_JOBS=4
ENV NVCC_THREADS=8

# Fix dynamic library search path
# Prioritize /usr/local/nvidia for cloud environments with host driver mounts
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Use system ptxas instead of Triton's built-in version
# Fix unsupported arch (e.g., GB10/sm_121a) in older Triton
# Safe: Triton falls back to built-in ptxas if path doesn't exist
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# -----------------------------------------------------------------------------
# 6. vLLM Precompiled Mode Config
# -----------------------------------------------------------------------------
# Enable precompiled mode: only compile Python changes, download official C++ wheels
ENV VLLM_USE_PRECOMPILED=1

# Official commit hash for v0.13.0 (from: git rev-parse HEAD)
ENV VLLM_PRECOMPILED_WHEEL_COMMIT=72506c98349d6bcd32b4e33eec7b5513453c1502

# CUDA version variant (cu12 recommended for compatibility)
ENV VLLM_PRECOMPILED_WHEEL_VARIANT=cu12

# -----------------------------------------------------------------------------
# 7. Workspace & Tools Config
# -----------------------------------------------------------------------------
WORKDIR /root/vllmbench

# Copy vllmbench source code into image
COPY . /root/vllmbench/

# Configure uv: increase timeout, use copy mode to avoid hardlink errors in Docker layers
ENV UV_HTTP_TIMEOUT=500
ENV UV_LINK_MODE=copy

CMD ["sleep", "infinity"]