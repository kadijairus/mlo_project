import torch
import pytest
from mlo_group_project.model import BreastCancerModel



#TO check if we are grabbing right parameters and HP
def test_optimizer_config():

    
    model = BreastCancerModel(input_shape=30)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    #optimizer track
    assert len(optimizer.param_groups) == 1
    #Param track
    assert len(optimizer.param_groups[0]['params']) > 0


#Training Loop check before running the hwole algorithm 
def test_one_training_step():
 
    #Setup
    model = BreastCancerModel(input_shape=30)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    #Fake Data
    inputs = torch.randn(4, 30)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0]).unsqueeze(1) # Shape (4, 1)

    #Forward
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    #Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Check if it worked
    assert loss.item() > 0
    # The model parameters should have gradients now (meaning it learned something)
    for param in model.parameters():
        assert param.grad is not None
