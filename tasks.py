import os

from invoke import Context, task
import subprocess

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlo_group_project"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def train_profile(ctx: Context) -> None:
    """Profile training."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py --profile", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)


@task
def visualize(ctx: Context) -> None:
    """Visualize model results."""
    ctx.run(f"uv run src/{PROJECT_NAME}/visualize.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", env={"PYTHONPATH": "."}, pty=not WINDOWS, echo=True)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )

@task
def serve_api(ctx: Context, port: int = 8000) -> None:
    """Serve FastAPI backend (opens in a new terminal on Windows)."""
    cmd = f"uv run uvicorn {PROJECT_NAME}.api:app --host 127.0.0.1 --port {port} --reload"

    if WINDOWS:
        full = f'{cmd} & echo. & echo API process exited. & pause'
        subprocess.Popen(
            ["cmd.exe", "/k", full],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    else:
        ctx.run(cmd, echo=True, pty=True)

@task
def serve_ui(ctx: Context, port: int = 8501) -> None:
    """Serve Streamlit UI (opens in a new terminal on Windows)."""
    cmd = (
        f"uv run streamlit run src/{PROJECT_NAME}/streamlit_app.py "
        f"--server.port {port}"
    )

    if WINDOWS:
        # /k keeps it open; pause shows errors if the command fails instantly
        full = f'{cmd} & echo. & echo UI process exited. & pause'
        subprocess.Popen(
            ["cmd.exe", "/k", full],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    else:
        ctx.run(cmd, echo=True, pty=True)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
