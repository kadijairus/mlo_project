from pathlib import Path

import typer
from torch.utils.data import Dataset
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class BreastCancerData(Dataset):
    """Breast Canser Wisconsis data set from: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data"""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        df = pd.read_csv(data_path)
        self.data = df

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        # Drop id column
        self.data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
        self.X = self.data.drop(columns=['diagnosis']) # M = malignant, B = benign
        self.y = self.data['diagnosis']
        scaler = MinMaxScaler() # max value = 1 , min value = 0 
        self.X = scaler.fit_transform(self.X)
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.9, shuffle=True, random_state=42)
        torch.save((torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32)), output_folder / "train.pt")
        torch.save((torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32)), output_folder / "test.pt")

    def normalize(self):
        pass

    def shuffle(self):
        """Shuffle the dataset."""
        indices = torch.randperm(self.images.shape[0])
        self.images = self.images[indices]
        self.targets = self.targets[indices]

def preprocess(data_path: Path = Path("src/mlo_group_project/data/raw/bcw.csv"), output_folder: Path = Path("src/mlo_group_project/data/processed")) -> None:
    print("Preprocessing data...")
    dataset = BreastCancerData(data_path)
    dataset.preprocess(output_folder)



if __name__ == "__main__":
    typer.run(preprocess)

 
