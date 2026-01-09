import torch
import typer
from pathlib import Path
from mlo_group_project.model import BreastCancerModel


#Setting up Typer app , For HP tuning 
app = typer.Typer()
@app.command()

def train_model(
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    processed_dir: Path = Path("data/processed"),
    model_save_path: Path = Path("models/model.pth"),
    
):
        print(f" Starting training: epochs={epochs}, lr={lr}, batch_size={batch_size}")
        print(f" Loading data from: {processed_dir}")
        



