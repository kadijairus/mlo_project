from __future__ import annotations
from pathlib import Path
from typing import Any
import os
import wandb

def init_wandb(project, config):
    mode = os.environ.get("WANDB_MODE", "offline")
    wandb.init(
        project=project,
        config=config,
        mode=mode
    )

def log_epoch(epoch: int, loss: float, acc: float) -> None:
    wandb.log({"epoch": epoch, "train_loss": loss, "train_acc": acc})

def log_model_artifact(path: Path, name: str, metadata: dict[str, Any]) -> None:
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)

def finish_wandb() -> None:
    if wandb.run is not None:
        wandb.finish()