import matplotlib.pyplot as plt
import torch
import typer
from pathlib import Path
from pic_classification_mnist_v01_xh.model import MyNeuralNet
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str = "models/model.pth", figure_name: str = "embeddings.png") -> None:
    """Visualize model embeddings using t-SNE.
    
    Args:
        model_checkpoint: Path to the saved model checkpoint
        figure_name: Name of the output figure file
    """
    print("Visualizing model embeddings...")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Using device: {DEVICE}")
    
    model: torch.nn.Module = MyNeuralNet().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()
    # Replace final classification layer with identity to get embeddings
    model.fc2 = torch.nn.Identity()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            images = images.to(DEVICE)
            predictions = model(images)
            embeddings.append(predictions.cpu())
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    print(f"Embedding shape: {embeddings.shape}")
    
    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        print("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i), alpha=0.6)
    plt.legend()
    plt.title("t-SNE visualization of model embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / figure_name
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    typer.run(visualize)
