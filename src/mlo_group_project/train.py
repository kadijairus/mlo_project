import torch
import typer
from pathlib import Path
from mlo_group_project.model import BreastCancerModel

# handling configurations and experimenting
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

#Setting up Typer app , For HP tuning 
app = typer.Typer()
@app.command()

@hydra.main(config_path="config", config_name="config", version_base=None)
def train_model(cnf: DictConfig):
    # Convert Hydra config to a standard dictionary
    wandb_config = OmegaConf.to_container(
        cnf, resolve=True, throw_on_missing=True
        )
    print(f" Starting training: epochs={cnf.hp.epochs}, lr={cnf.hp.lr}, batch_size={cnf.hp.batch_size}")
    print(f" Loading data from: {cnf.paths.processed_dir}")
    print(f" Save model to: {cnf.paths.model_save_path}")
        
if __name__  == "__main__":
    train_model()
