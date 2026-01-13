import torch
from mlo_group_project.model import BreastCancerModel


#Sanity check on eval mode. if model is on eval, it should give back the same answer for the same input. 
def test_model_is_deterministic():
    
    model = BreastCancerModel(input_shape=30)
  
    #Model on eval  
    model.eval()  

    #Random input
    input_data = torch.randn(5, 30)
    
    # Run it twice
    output1 = model(input_data)
    output2 = model(input_data)
    
    # Check if they are identical
    assert torch.allclose(output1, output2), "Model is acting random when it should be stable!"


#Check if we change batch sizes model crashes or still runs ( Different devices can utulize different batch sizes depending on processing power and memory )
def test_batch_size_flexibility():
    
    model = BreastCancerModel(input_shape=30)
    
    #Single patient
    out_single = model(torch.randn(1, 30))
    assert out_single.shape == (1, 1)
    
    #Huge batch
    out_huge = model(torch.randn(128, 30))
    assert out_huge.shape == (128, 1)
