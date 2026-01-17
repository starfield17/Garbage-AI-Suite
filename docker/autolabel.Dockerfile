# ==============================================================================
# AutoLabel Context Dockerfile
# ==============================================================================

# Build stage
FROM python:3.10-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY pyproject.toml autolabel/pyproject.toml ./
COPY shared/ shared/

WORKDIR /tmp/autolabel
RUN pip install --no-cache-dir \
    ultralytics openai opencv-python tqdm torch torchvision \
    && pip install --no-cache-dir --no-deps \
    -e /tmp/shared \
    -e .

# Runtime stage
FROM python:3.10-slim-bookworm AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /data

# Default command
CMD ["autolabel"]
