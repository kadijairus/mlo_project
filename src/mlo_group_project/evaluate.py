from loguru import logger
import torch
import typer
from pathlib import Path
from mlo_group_project.model import BreastCancerModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer()

@app.command()

#Evaluate the trained model on the test dataset
def evaluate_model(
        model_path: Path = Path("models/model.pth"),
        processed_dir: Path = Path("data/processed"),
        metrics_save_path: Path = Path("eval_metrics.pt")
):
    """Evaluate the trained model on the test dataset."""
    try:
        logger.info(f"Starting evaluation using model: {model_path}")

        # --- Enhanced Data Loading ---
        eval_file = processed_dir / "test.pt"
        try:
            logger.debug(f"Loading evaluation data from: {eval_file}")
            data = torch.load(eval_file)
            x_eval, y_eval = data[0], data[1]
        except FileNotFoundError:
            logger.critical(f"Evaluation data not found at '{eval_file}'. Please run the preprocessing script.")
            raise
        except Exception as e:
            logger.critical(f"Failed to load or parse evaluation data from '{eval_file}'. Error: {e}")
            raise

        # --- Data Validation ---
        assert x_eval.shape[0] == y_eval.shape[0], \
            f"Shape mismatch between evaluation inputs and targets: {x_eval.shape[0]} != {y_eval.shape[0]}"
        logger.debug(f"Evaluation data shapes - x: {x_eval.shape}, y: {y_eval.shape}")

        # --- Enhanced Model Loading ---
        try:
            model = BreastCancerModel(input_shape=x_eval.shape[1])
            model.load_state_dict(torch.load(model_path))
            model.eval()
            logger.debug("Model loaded and set to evaluation mode.")
        except FileNotFoundError:
            logger.critical(f"Trained model not found at '{model_path}'. Please run the training script.")
            raise
        except RuntimeError as e:
            logger.critical(
                "Failed to load model state. This often means the model architecture in the script "
                f"does not match the architecture saved in '{model_path}'.\nOriginal Error: {e}"
            )
            raise

        # --- Evaluation Logic ---
        logger.info("Performing evaluation...")
        with torch.no_grad():
            logits = model(x_eval)
            probabilities = torch.sigmoid(logits.squeeze())
            predicted_classes = (probabilities > 0.5).float()
            y_eval = y_eval.float()
            logger.debug("Predictions computed.")

            correct = (predicted_classes == y_eval.view_as(predicted_classes)).sum().item()
            total = y_eval.size(0)

        if total == 0:
            logger.warning("Evaluation dataset is empty. Accuracy cannot be calculated.")
            accuracy = 0.0
        else:
            accuracy = correct / total

        # --- Log Final Report ---
        logger.info(f"\n{'=' * 30}"
                    f"\n FINAL EVALUATION REPORT"
                    f"\n   Total Samples: {total}"
                    f"\n   Correct:       {correct}"
                    f"\n   Accuracy:      {accuracy * 100:.2f}%"
                    f"\n{'=' * 30}\n")

        # --- Enhanced Metrics Saving ---
        try:
            metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "targets": y_eval,
                "predictions": predicted_classes,
                "probabilities": probabilities,
                "accuracy": accuracy
            }, metrics_save_path)
            logger.info(f"Evaluation metrics saved to: {metrics_save_path}")
        except OSError as e:
            logger.error(f"Could not save evaluation metrics to '{metrics_save_path}'. Check permissions. Error: {e}")
            # We don't re-raise here because the evaluation itself was successful.
            # Failing to save metrics is an error, but not as critical as failing the evaluation.

        logger.success("Evaluation script finished successfully.")

    except Exception as e:
        logger.critical(f"An unrecoverable error occurred during the evaluation process: {e}")
        # Use typer.Exit to ensure the script terminates with a non-zero exit code,
        # which is crucial for CI/CD pipelines.
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
