# Use Astral UV base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build tools (needed for some ML libraries)
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the project name as a build argument or env var
ARG PROJECT_NAME=mlo_group_project
ENV PROJECT_NAME=${PROJECT_NAME}

# Set working directory
WORKDIR /app

# 1. Install dependencies first (better caching)
# Copy only the requirements file
COPY dockerfiles/eval_requirements.txt ./requirements.txt

# Use uv for faster installation
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -r requirements.txt

# Copy metadata first (helps with caching)
COPY pyproject.toml README.md* ./
# Copy the rest of the application
COPY tasks.py ./tasks.py
# Copy the whole src directory to maintain structure
COPY src/${PROJECT_NAME}/api.py ./src/${PROJECT_NAME}/api.py
COPY src/${PROJECT_NAME}/streamlit_app.py ./src/${PROJECT_NAME}/streamlit_app.py
COPY src/${PROJECT_NAME}/styles/ ./src/${PROJECT_NAME}/styles/
COPY src/${PROJECT_NAME}/model.py ./src/${PROJECT_NAME}/model.py
COPY models/model.pth ./models/model.pth
COPY data/processed/scaler.joblib ./data/processed/scaler.joblib
COPY data/processed/feature_columns.json ./data/processed/feature_columns.json
COPY data/processed/label_encoder.joblib ./data/processed/label_encoder.joblib


# Copy git metadata
COPY .git ./.git
COPY .gitignore ./.gitignore

# Ensure the package directory exists
RUN mkdir -p src/${PROJECT_NAME} && touch src/${PROJECT_NAME}/__init__.py
ENTRYPOINT ["uv", "run", "--no-project", "invoke"]
CMD ["serve-api-ui"]
