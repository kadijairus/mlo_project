import pytest
import torch
from mlo_group_project.model import BreastCancerModel


@pytest.mark.parametrize("input_tensor", [
    torch.zeros(10, 30),        #All zeros
    torch.ones(10, 30),         #All ones
    torch.randn(10, 30) * 100,  #Huge random numbers
    torch.full((10, 30), -1.0)  #Negative numbers
])
def test_model_robustness_to_weird_values(input_tensor):
    
    # Init model
    model = BreastCancerModel(input_shape=30)
    
    # Run the model
    output = model(input_tensor)
    
    #Check if model can handle it 
    assert not torch.isnan(output).any(), "Model output contained NaNs!"
    
    
    assert not torch.isinf(output).any(), "Model output contained Infinity!"
    
    
    assert output.shape == (10, 1), f"Expected shape (10, 1), got {output.shape}"
