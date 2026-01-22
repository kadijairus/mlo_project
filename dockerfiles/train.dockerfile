# Use Astral UV base image with Python 3.12 on Debian Bookworm slim
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
# Install build tools
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
# Install dependencies
COPY ../uv.lock uv.lock
COPY ../pyproject.toml pyproject.toml
COPY ../README.md README.md
COPY ../src src/
# Set working directory to root
#WORKDIR /
WORKDIR /app

# Install dependencies from uv.lock without installing the project itself
RUN uv sync --locked --no-cache --no-install-project

# Set the entrypoint to run the training script
ENTRYPOINT ["uv", "run", "src/mlo_group_project/train.py"]
