from loguru import logger
from omegaconf import DictConfig
import torch.nn as nn
import torch


# Simple ANN with 30 input nodes, all activations are ReLU for simplicity, Having some dropout to counter overfit
class BreastCancerModel(nn.Module):
    def __init__(self, input_shape: int, cnf: DictConfig | None = None) -> None:
        super(BreastCancerModel, self).__init__()
        logger.info(f"Initializing BreastCancerModel with input shape: {input_shape}")
        seed = 42 if cnf is None else cnf.hp.seed
        torch.manual_seed(int(seed))
        if not isinstance(input_shape, int) or input_shape <= 0:
            logger.critical(
                f"Invalid input_shape: '{input_shape}'. The number of input features must be a positive integer."
            )
            raise ValueError("Model input_shape must be a positive integer.")

        self.network = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
        logger.success("BreastCancerModel initialized successfully.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            output: torch.Tensor = self.network(x)
            return output
        except RuntimeError as e:
            expected_features = self.network[0].in_features
            logger.critical(
                f"CRITICAL ERROR during forward pass. Likely input shape mismatch."
                f"\n  - Model's first layer expects an input with {expected_features} features."
                f"\n  - The actual input tensor provided has shape: {x.shape}."
                f"\n  - Original PyTorch Error: {e}"
            )
            raise
