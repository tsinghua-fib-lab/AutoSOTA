# Docker environment

Reproducible build of the eb-vllm runtime. Mirrors the environment used
for the paper experiments.

## Build

```bash
./docker-build.sh
# or directly
docker build -t eb-vllm:icml2026 -f Dockerfile ..
```

The build context is the repo root (so the Dockerfile can `COPY` the vLLM
source and the `reproduce/` tree).

## Run

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/outputs:/workspace/reproduce/synthetic_e2e/outputs \
    eb-vllm:icml2026
```

Outputs from any `run_*.sh` are written under each `sec*/outputs/` —
mount whichever ones you care about.
