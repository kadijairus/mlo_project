import torch
from torch import nn


#Model : All activations are ReLU , Dropout to combat overfitting ,
# 30 neurons in input layer ,
class BreastCancerModel(nn.Module):
    def __init__(self, input_shape: int = 30):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.network(x)   
