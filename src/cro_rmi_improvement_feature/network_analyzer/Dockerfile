# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBER_OF_WORKER=32

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
COPY src/cro_rmi_improvement_feature/network_analyzer/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY src ./src

# Ensure src on PYTHONPATH for test discovery/imports
ENV PYTHONPATH=/app/src

# Run tests at build-time to validate image (optional; comment out to speed up)
RUN pytest -q || (RC=$?; echo "Tests failed with exit code $RC"; exit $RC)

EXPOSE 7900

# Default entrypoint to run the FastAPI app
CMD ["python", "api/main.py"]

