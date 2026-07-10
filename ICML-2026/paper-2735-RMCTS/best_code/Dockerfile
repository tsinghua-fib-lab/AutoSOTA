FROM nvcr.io/nvidia/tensorrt:24.08-py3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    onnx \
    onnxruntime-gpu \
    pycuda \
    torch --extra-index-url https://download.pytorch.org/whl/cu124

# Set working directory
WORKDIR /app

# Copy project files (when building)
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
