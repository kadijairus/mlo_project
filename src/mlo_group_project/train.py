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
    processed_dir: Path = Path("src/mlo_group_project/data/processed"),
    model_save_path: Path = Path("src/mlo_group_project/models/model.pth"),
    metrics_save_path: Path = Path("src/mlo_group_project/reports/metrics.pt")
    
):
        print(f" Starting training: epochs={epochs}, lr={lr}, batch_size={batch_size}")
        print(f" Loading data from: {processed_dir}")
        data_file = processed_dir / "train.pt"
        print(f" Loading data from: {data_file}")
        data = torch.load(data_file)
        X_train = data["images"]
        y_train = data["targets"]

        #Sanity check
        print("\n Data Sanity Check:")
        print(f"   Input Shape (X): {X_train.shape}")
        print(f"   Target Shape (y): {y_train.shape}")
        print(f"   First 2 Targets: {y_train[:2].tolist()}")
        print("-" * 20 + "\n")

        #Loading data into DataLoader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        #Model Initialization, Forcing input shape based on data
        input_features = X_train.shape[1]
        model = BreastCancerModel(input_shape=input_features)

        #Loss function and optimizer, We use BCEWithLogitsLoss for binary classification
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        #Some metrics to track
        loss_history = []
        accuracy_history = []

        #Train
        print(" Starting training:")
        model.train()

        #Training Loop
        for epoch in range(epochs):
                epoch_loss = 0.0
                correct = 0
                total = 0
                batch_losses = []

                for batch_X, batch_y in dataloader:
                        optimizer.zero_grad()
                        predictions = model(batch_X)

                        #Misshape fix
                        loss = criterion(predictions.squeeze(), batch_y.float())

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        probs = torch.sigmoid(predictions.squeeze())
                        preds = (probs > 0.5).float()
                        correct += (preds == batch_y).sum().item()
                        total += batch_y.size(0)
                        batch_losses.append(loss.item())

                #Average metrics
                avg_loss = epoch_loss / len(dataloader)
                avg_acc = correct / total

                #Store metrics
                loss_history.append(avg_loss)
                accuracy_history.append(avg_acc)

                #Report epoch loss every 10 epochs
                if (epoch + 1) % 10 == 0:
                        avg_loss = sum(batch_losses) / len(batch_losses)
                        print(f"   Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        #Save the trained model    
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f" Model saved successfully to: {model_save_path}")

        #Save metrics
        metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "loss": loss_history, 
            "accuracy": accuracy_history,
            "epochs": list(range(1, epochs + 1))
        }, metrics_save_path)
        print(f" Metrics saved for plotting to: {metrics_save_path}")


if __name__ == "__main__":
    app()                    




