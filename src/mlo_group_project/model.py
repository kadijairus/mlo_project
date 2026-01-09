import torch
import torch.nn as nn


#We just use a simple ANN , with 30 input nodes , all activations are ReLU for simplicity, Having some dropout to counter overfit
class BreastCancerModel(nn.Module):
    def __init__(self, input_shape: int):
        super(BreastCancerModel, self).__init__()
        
        
        self.network = nn.Sequential(
            
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  
            
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
