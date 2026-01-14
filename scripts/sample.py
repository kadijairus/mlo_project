import torch
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def create_sample_data():
    """Create a small sample dataset from the full training data for testing purposes."""
    full_data_path = PROJECT_ROOT / "data" / "processed" / "train.pt"
    save_path = PROJECT_ROOT / "tests" / "sample_data.pt"

    if not full_data_path.exists():
        logger.error(f"! {full_data_path} not found. Run data.py first!")
        return

    x, y = torch.load(full_data_path)

    #define sampling
    sample_x = x[:5]
    sample_y = y[:5]

    #Save sample
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((sample_x, sample_y), save_path)

    logger.success(f"Created {save_path} with shape: {sample_x.shape}")


if __name__ == "__main__":
    create_sample_data()
