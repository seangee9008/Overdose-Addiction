# Container for the Python pipeline. Data are NOT baked in: mount the repo's data/ (raw downloads)
# and it runs `make all`. Without a mount it can still run the synthetic-fixture CI checks:
#   docker build -t overdose-pipeline .
#   docker run --rm overdose-pipeline make ci PY=python DBT=dbt
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" -v "$PWD/results:/app/results" overdose-pipeline
# R is not installed, so Step C's optional R svyglm cross-check is skipped inside the container.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.0

WORKDIR /app
COPY requirements.txt requirements.lock ./
# Pipeline packages only (no JupyterLab / geopandas), pinned to the exact versions in requirements.lock.
RUN grep -vE '^(jupyterlab|geopandas)$' requirements.txt > /tmp/requirements-pipeline.txt \
    && uv pip install --system --no-cache -r /tmp/requirements-pipeline.txt -c requirements.lock

COPY . .
CMD ["make", "all", "PY=python", "DBT=dbt"]
