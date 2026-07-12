# Dockerfile for the eb-vllm reproduction harness.
#
# Build (from repo root):
#   docker build -t eb-vllm:icml2026 -f reproduce/docker/Dockerfile .
#
# Base image: vllm-openai built from this fork (release/icml2026), so the
# EB scheduler is already compiled in.

ARG BASE_IMAGE=vllm-openai
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir matplotlib numpy transformers datasets quart

COPY reproduce/ /workspace/reproduce/
COPY benchmarks/ /workspace/benchmarks/

RUN chmod +x /workspace/reproduce/**/*.sh 2>/dev/null || true

WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]
