FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

ARG TORCH_DEVICE=cpu
ENV UV_PROJECT_ENVIRONMENT=/venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --extra ${TORCH_DEVICE}

ENV PATH=/venv/bin:$PATH
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

COPY . .

# Editable install is critical: ROOT_DIR resolves to /app, not site-packages
RUN pip install -e . --no-deps

RUN mkdir -p /app/models /app/output /app/logs /app/data

ENTRYPOINT ["python", "-m", "moretro.moretro_star"]
