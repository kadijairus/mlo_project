from loguru import logger
from pathlib import Path
import typer
from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class BreastCancerData(Dataset):
    """Breast Canser Wisconsis data set from: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data"""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        try:
            df = pd.read_csv(self.data_path)
            logger.debug(f"Data read from csv. Data shape before preprocessing: {df.shape}")
            self.data = df
        except FileNotFoundError:
            logger.error(f"The file at {self.data_path} was not found.")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred while reading {self.data_path}: {e}")
            raise
        self.data = df

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        try:
            # Drop id column
            self.data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
            self.X = self.data.drop(columns=['diagnosis'])  # M = malignant, B = benign
            self.y = self.data['diagnosis']
            logger.info("Successfully dropped unnecessary columns and separated features and target.")
        except KeyError as e:
            logger.error(f"A required column was not found in the dataset: {e}")
            logger.debug(f"Available columns are: {self.data.columns.tolist()}")
            raise
        scaler = MinMaxScaler() # max value = 1 , min value = 0 
        self.X = scaler.fit_transform(self.X)
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.9, shuffle=True, random_state=42)

        try:
            # Ensure the output directory exists
            output_folder.mkdir(parents=True, exist_ok=True)

            train_path = output_folder / "train.pt"
            test_path = output_folder / "test.pt"

            torch.save(
                (torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32)),
                train_path)
            logger.info(f"Training data saved to {train_path}")

            torch.save((torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32)),
                       test_path)
            logger.info(f"Test data saved to {test_path}")

        except OSError as e:
            logger.error(f"Could not save processed data to {output_folder}. Check permissions or path. Error: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred while saving the processed data: {e}")
            raise

    def normalize(self):
        pass

    def shuffle(self):
        """Shuffle the dataset."""
        indices = torch.randperm(self.images.shape[0])
        self.images = self.images[indices]
        self.targets = self.targets[indices]

def preprocess(data_path: Path = Path("./data/raw/bcw.csv"), output_folder: Path = Path("./data/processed")) -> None:
    try:
        logger.debug(f"Preprocessing data from {data_path}...")
        dataset = BreastCancerData(data_path)
        dataset.preprocess(output_folder)
        logger.success(f"Data successfully preprocessed and saved to {output_folder}") # Use logger.success for a nice final message
    except Exception as e:
        logger.critical(f"The preprocessing script failed. Error: {e}")


if __name__ == "__main__":
    typer.run(preprocess)
