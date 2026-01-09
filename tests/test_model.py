import torch
from mlo_group_project.model import BreastCancerModel


#Dummy set up , for testing
def test_model_structure():

    #Size check
    input_features = 30
    model = BreastCancerModel(input_shape=input_features)
    
    # Create a dummy input 
    dummy_input = torch.randn(1, input_features)
    
    # Push it through the model
    output = model(dummy_input)
    
    # Check if output is exactly 1 number (raw prediction)
    assert output.shape == (1, 1)
