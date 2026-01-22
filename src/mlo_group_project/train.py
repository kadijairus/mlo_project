from __future__ import annotations
import os
from loguru import logger
import sys
from pathlib import Path
from torch import optim, nn
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, cast
from dotenv import load_dotenv

from mlo_group_project.model import BreastCancerModel
from mlo_group_project.training.checkpoints import (
    CheckpointState,
    save_state_dict,
    update_best,
)
from mlo_group_project.training.data import load_train_tensors, make_train_dataloader
from mlo_group_project.training.loop import fit, EpochMetrics
from mlo_group_project.training.utils import save_metrics
from mlo_group_project.training.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_epoch,
    log_model_artifact,
)

load_dotenv()

@hydra.main(config_path="config", config_name="config", version_base=None)
def train_model(cnf: DictConfig) -> None:
    """Train the Breast Cancer Classification Model."""
    try:
        wandb_config = cast(
            dict[str, Any],
            OmegaConf.to_container(cnf, resolve=True, throw_on_missing=True),
        )
        init_wandb(project="Breast Cancer Wisconsin", config=wandb_config)

        epochs = int(cnf.hp.epochs)
        lr = float(cnf.hp.lr)
        batch_size = int(cnf.hp.batch_size)

        processed_dir = Path(cnf.paths.processed_dir)
        model_save_path = Path(cnf.paths.model_save_path)
        metrics_save_path = Path(cnf.paths.metrics_save_path)

        best_ckpt_path = model_save_path.parent / "best_model.pt"
        last_ckpt_path = model_save_path.parent / "last_model.pt"

        data = load_train_tensors(processed_dir)
        dataloader = make_train_dataloader(data, batch_size=batch_size, shuffle=True)

        input_features = int(data.x.shape[1])
        model = BreastCancerModel(input_shape=input_features, cnf=cnf)

        criterion: nn.Module = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        ckpt_state = CheckpointState()

        def on_epoch_end(epoch: int, metrics: EpochMetrics) -> None:
            log_epoch(epoch=epoch, loss=metrics.loss, acc=metrics.acc)
            update_best(
                state=ckpt_state,
                model=model,
                metric_value=metrics.loss,
                epoch=epoch,
                best_path=best_ckpt_path,
                lower_is_better=True,
            )

        loss_hist, acc_hist = fit(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            on_epoch_end=on_epoch_end,
        )

        # Save final artifacts locally
        save_state_dict(model, model_save_path)
        save_state_dict(model, last_ckpt_path)
        save_metrics(metrics_save_path, loss=loss_hist, acc=acc_hist)

        # Log W&B artifacts
        log_model_artifact(
            last_ckpt_path,
            name="breast-cancer-model-last",
            metadata={
                "epoch": epochs,
                "train_loss_last": float(loss_hist[-1]),
                "train_acc_last": float(acc_hist[-1]),
                "lr": lr,
                "batch_size": batch_size,
            },
        )

        if ckpt_state.best_metric is not None and ckpt_state.best_epoch is not None:
            log_model_artifact(
                best_ckpt_path,
                name="breast-cancer-model-best",
                metadata={
                    "epoch": ckpt_state.best_epoch,
                    "train_loss_best": float(ckpt_state.best_metric),
                    "lr": lr,
                    "batch_size": batch_size,
                },
            )

        logger.success("Training process finished successfully!")

    except Exception as e:
        logger.critical(f"A critical error occurred during the training process: {e}")
        # Re-raise the exception to ensure the script exits with a non-zero code
        raise
    finally:
        finish_wandb()


if __name__ == "__main__":
    # Check if profiling is requested
    if "--profile" in sys.argv:
        import cProfile

        # Remove --profile from sys.argv so Hydra doesn't see it
        sys.argv.remove("--profile")
        logger.info("Profiling enabled")
        profiler = cProfile.Profile()
        profiler.enable()
        train_model()
        profiler.disable()
        profiler.dump_stats("reports/train_profile.prof")
        logger.info(
            "Profile saved to reports/train_profile.prof\n To visualize, run: snakeviz reports/train_profile.prof"
        )
    else:
        train_model()
