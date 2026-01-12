from loguru import logger
import torch
from pathlib import Path
import math
from mlo_group_project.model import BreastCancerModel
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from typing import Any, cast
from dotenv import load_dotenv
load_dotenv()
import cProfile
import sys

@hydra.main(config_path="config", config_name="config", version_base=None)
def train_model(cnf: DictConfig):
    """Train the Breast Cancer Classification Model."""
    try:
        # Convert Hydra config to a standard dictionary
        wandb_config = cast(dict[str, Any], 
                            OmegaConf.to_container(
                                cnf, resolve=True, throw_on_missing=True
                            ))
        wandb.init(project="Breast Cancer Wisconsin", config=wandb_config)
        logger.info("Starting model training process...")
        logger.debug(f"Configuration loaded: {OmegaConf.to_yaml(cnf)}")

        # Extract variables from Hydra Config (M11)
        epochs = cnf.hp.epochs
        lr = cnf.hp.lr
        batch_size = cnf.hp.batch_size
        logger.debug(f"Hyperparameters - epochs: {epochs}, lr: {lr}, batch_size: {batch_size}")

        processed_dir = Path(cnf.paths.processed_dir)
        model_save_path = Path(cnf.paths.model_save_path)
        metrics_save_path = Path(cnf.paths.metrics_save_path)

        # --- Enhanced Data Loading ---
        logger.debug(f"Getting files from: {processed_dir}")
        data_file = processed_dir / "train.pt"
        try:
            logger.debug(f"Loading data from file: {data_file}")
            x_train, y_train = torch.load(data_file)
        except FileNotFoundError:
            logger.error(f"Training data not found at {data_file}. Please run the preprocessing script first.")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred while loading the data: {e}")
            raise

        # --- Enhanced Sanity Check ---
        assert x_train.shape[0] == y_train.shape[0], \
            f"Shape mismatch between inputs and targets: {x_train.shape[0]} != {y_train.shape[0]}"

        logger.info(f"\n{'=' * 10} Data Sanity Check {'=' * 10}"
                    f"\n   Input Shape (x): {x_train.shape}"
                    f"\n   Target Shape (y): {y_train.shape}"
                    f"\n   First 2 Targets: {y_train[:2].tolist()}"
                    f"\n{'=' * 39}\n")

        # Load data into DataLoader
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Model Initialization, Forcing input shape based on data
        input_features = x_train.shape[1]
        logger.debug(f"Initializing model with input features: {input_features}")
        model = BreastCancerModel(input_shape=input_features)

        # Loss function and optimizer, We use BCEWithLogitsLoss for binary classification
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Some metrics to track
        loss_history = []
        accuracy_history = []

        logger.info("Starting training loop...")
        model.train()

        # Training Loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y.float())

                # --- Numerical Stability Check ---
                if not math.isfinite(loss.item()):
                    logger.error(f"Loss is {loss.item()} at epoch {epoch + 1}, stopping training.")
                    raise ValueError("Non-finite loss detected")

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Calculate accuracy
                with torch.no_grad():
                    logits = predictions.squeeze()
                    preds = (logits > 0).to(batch_y.dtype)
                    correct += (preds == batch_y).sum().item()
                    total += batch_y.size(0)

            # End of batch loop. Average metrics.
            avg_loss = epoch_loss / len(dataloader)
            avg_acc = correct / total
            logger.debug(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc * 100:.2f}%")

            # Store metrics
            loss_history.append(avg_loss)
            accuracy_history.append(avg_acc)

            # Log to Wandb
            wandb.log({"train_loss": avg_loss, "train_acc": avg_acc, "epoch": epoch})

        logger.info("Training loop completed.")

        # --- Enhanced Artifact Saving ---
        try:
            # Save the trained model
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.debug(f"Model saved successfully to: {model_save_path}")

            # Save metrics
            metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "loss": loss_history,
                "accuracy": accuracy_history,
                "epochs": list(range(1, epochs + 1))
            }, metrics_save_path)
            logger.debug(f"Metrics saved for plotting to: {metrics_save_path}")
        except OSError as e:
            logger.error(f"Failed to save artifacts. Check permissions for the path. Error: {e}")
            raise

        logger.success("Training process finished successfully!")

    except Exception as e:
        logger.critical(f"A critical error occurred during the training process: {e}")
        # Re-raise the exception to ensure the script exits with a non-zero code
        raise

    finally:
        # --- Ensure wandb is always closed ---
        if wandb.run is not None:
            logger.info("Closing wandb run.")
            wandb.finish()


if __name__ == "__main__":
    # Check if profiling is requested
    if "--profile" in sys.argv:
        # Remove --profile from sys.argv so Hydra doesn't see it
        sys.argv.remove("--profile")
        logger.info("Profiling enabled")
        profiler = cProfile.Profile()
        profiler.enable()
        train_model()
        profiler.disable()
        profiler.dump_stats("reports/train_profile.prof")
        logger.info("Profile saved to reports/train_profile.prof\n To visualize, run: snakeviz reports/train_profile.prof")
    else:
        train_model()
