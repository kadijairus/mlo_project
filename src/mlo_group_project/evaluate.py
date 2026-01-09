import torch
import typer
from pathlib import Path
from mlo_group_project.model import BreastCancerModel

app = typer.Typer()

@app.command()

#Evaluate the trained model on the test dataset
def evaluate_model(
    model_path: Path = Path("models/model.pth"),
    processed_dir: Path = Path("data/processed"),
    metrics_save_path: Path = Path("reports/eval_metrics.pt")
):
    
    print(f" Starting evaluation using model: {model_path}")

    eval_file = processed_dir / "test.pt"
    print(f" Loading eval data from: {eval_file}")
    
    data = torch.load(eval_file)
    X_eval = data["images"]
    y_eval = data["targets"]

    # Load the trained model
    model = BreastCancerModel(input_shape=X_eval.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()


    # Perform evaluation , No grad cause we are not training
    with torch.no_grad():
        logits = model(X_eval)
        probabilities = torch.sigmoid(logits.squeeze())
        predicted_classes = (probabilities > 0.5).float()
        
        correct = (predicted_classes == y_eval).sum().item()
        total = y_eval.size(0)
    # Calculate accuracy
    accuracy = correct / total


    # Print evaluation metrics
    print("\n" + "=" * 30)
    print(f" FINAL EVALUATION REPORT")
    print(f"   Total Samples: {total}")
    print(f"   Correct:       {correct}")
    print(f"   Accuracy:      {accuracy * 100:.2f}%")
    print("=" * 30 + "\n")

    # Save evaluation metrics
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "targets": y_eval,             
        "predictions": predicted_classes, 
        "probabilities": probabilities,   
        "accuracy": accuracy
    }, metrics_save_path)
    
    # Log the location of saved metrics
    print(f" Eval metrics saved to: {metrics_save_path}")

if __name__ == "__main__":
    app()

    
