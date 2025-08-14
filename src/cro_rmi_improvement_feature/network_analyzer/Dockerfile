# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# System dependencies for scientific/graph packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libcairo2 \
    libcairo2-dev \
    libxml2 \
    libxml2-dev \
    zlib1g-dev \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency descriptors first to leverage Docker layer cache
COPY pyproject.toml ./
COPY ./pyproject.toml ./pyproject.toml

RUN python -m pip install --upgrade pip

# Install Python dependencies via requirements.txt
COPY ./requirements.txt ./requirements.txt
RUN  pip install -r  requirements.txt

# Copy source code
COPY data_processor ./data_processor
COPY api ./api
COPY core ./core
COPY config ./config
# create folder data/graph
RUN mkdir -p data/graph
RUN mkdir -p data/raw
COPY data/graph/graph_data_library.pkl ./data/graph/graph_data_library.pkl
COPY data/raw/Stakeholder_Analysis_WJMod.xlsx ./data/raw/Stakeholder_Analysis_WJMod.xlsx
# Ensure src on PYTHONPATH for test discovery/imports
ENV PYTHONPATH=/app/src

# Run tests at build-time to validate image (optional; comment out to speed up)
# RUN pytest -q || (RC=$?; echo "Tests failed with exit code $RC"; exit $RC)

EXPOSE $PORT

# Default entrypoint to run the FastAPI app
CMD ["sh", "api/run_api.sh"]

