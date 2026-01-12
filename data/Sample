import torch
from pathlib import Path


full_data_path = Path("data/processed/train.pt")
X, y = torch.load(full_data_path)

#define sampling 
sample_X = X[:5]
sample_y = y[:5]

#Save sample
save_path = Path("tests/sample_data.pt")
torch.save((sample_X, sample_y), save_path)

print(f"Created {save_path} with shape: {sample_X.shape}")
