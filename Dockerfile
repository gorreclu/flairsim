# FlairSim -- Drone benchmark over FLAIR-HUB aerial imagery.
# Multi-stage build: install dependencies, then copy app code.

FROM python:3.11-slim AS base

# System dependencies for rasterio (GDAL).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgdal-dev \
        gdal-bin \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package (server extras).
COPY pyproject.toml README.md ./
COPY flairsim/ flairsim/
RUN pip install --no-cache-dir ".[server]"

# Copy scenario definitions.
COPY scenarios/ scenarios/

# Data and leaderboard DB are mounted at runtime (not baked in).
# VOLUME /app/data

EXPOSE 8000

# Launch the orchestrator.
CMD ["flairsim-web", "--host", "0.0.0.0", "--port", "8000", "--scenarios-dir", "/app/scenarios", "--data-root", "/app/data"]
