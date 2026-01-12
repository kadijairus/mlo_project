from pathlib import Path

from torch.utils.data import Dataset
import os
from mlo_group_project.data import BreastCancerData


def test_my_dataset():
    """Test the MyDataset class."""
    print(f"\nPytest is running from: {os.getcwd()}")
    dataset = BreastCancerData(Path("../data/raw/bcw.csv"))
    assert isinstance(dataset, Dataset)
