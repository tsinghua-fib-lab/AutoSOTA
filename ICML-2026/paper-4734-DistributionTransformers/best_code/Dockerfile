FROM python:3.12 AS base

WORKDIR /app

COPY pyproject.toml .
RUN pip install '.[dev]'

FROM base AS app
COPY . /app

RUN pip install .

CMD ["python", "main.py"]
