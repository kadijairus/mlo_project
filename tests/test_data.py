from torch.utils.data import Dataset
import os
from mlo_group_project.data import BreastCancerData
import torch
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def test_data_dataset_is_created_from_raw_data():
    logger.debug(f"\nPytest is running from: {os.getcwd()}")
    raw_path = PROJECT_ROOT / "data" / "raw" / "bcw.csv"
    dataset = BreastCancerData(Path(raw_path))
    assert isinstance(dataset, Dataset)


# Data structure check
def test_data_sample_data_structure_correct():
    data_path = PROJECT_ROOT / "tests" / "sample_data.pt"
    assert data_path.exists(), f"Sample data missing at {data_path}"

    images, targets = torch.load(data_path)

    # Shape check
    assert images.ndim == 2
    assert images.shape == (5, 30)
    assert images.shape[1] == 30
    assert images.shape[0] == targets.shape[0]
    assert targets.shape == (5,)

    # Type check
    assert isinstance(images, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
