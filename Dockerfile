FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
# libgomp1 and build-essential already included in cuda image
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy Poetry configuration files
COPY ./pyproject.toml ./poetry.lock ./README.md ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Copy application source
COPY ./src ./src

# Install the ModelMora package itself
RUN poetry install --only-root --no-interaction --no-ansi

# Expose ports
# 8080 for REST API
EXPOSE 8080
# 50051 for gRPC
EXPOSE 50051

CMD ["python3.10", "-m", "ModelMora.main"]
