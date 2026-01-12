import torch
import pytest
from pathlib import Path

import mlo_group_project 


#Data structure check 
def test_data_structure():
    #Load directly
    data_path = Path(__file__).parent / "sample_data.pt"
    images, targets = torch.load(data_path)

    #Shape check
    assert images.shape == (10, 30)
    assert targets.shape == (10,)
    
    #Type check
    assert isinstance(images, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
