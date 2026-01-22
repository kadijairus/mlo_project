from pathlib import Path
import matplotlib.pyplot as plt
import torch
import typer
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from pic_classification_mnist_v01_xh.model import MyNeuralNet


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def visualize_embeddings(
    model_checkpoint: str,
    method: str = "umap",   # umap | tsne
    figure_name: str = "embeddings.png",
):
    print("Visualizing ANN embeddings")
    print(f"Method: {method}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = MyNeuralNet().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()
    model.fc2 = torch.nn.Identity()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    images = torch.load("data/processed/test_images.pt")
    labels = torch.load("data/processed/test_target.pt")

    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    Z, y = [], []
    with torch.inference_mode():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            z = model(x_batch)
            Z.append(z.cpu())
            y.append(y_batch)

    Z = torch.cat(Z).numpy()
    y = torch.cat(y).numpy()

    print("Embedding shape:", Z.shape)

    # ------------------------------------------------------------------
    # Optional PCA
    # ------------------------------------------------------------------
    if Z.shape[1] > 100:
        Z = PCA(n_components=50).fit_transform(Z)

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        Z_2d = reducer.fit_transform(Z)

    elif method == "umap":
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )
        Z_2d = reducer.fit_transform(Z)

    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    # ------------------------------------------------------------------
    # Plot with class-conditional colorbar
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 9))

    scatter = plt.scatter(
        Z_2d[:, 0],
        Z_2d[:, 1],
        c=y,
        cmap="tab10",
        s=15,
        alpha=0.8,
    )

    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.set_label("Class label")

    plt.title(f"ANN Embeddings ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / figure_name

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    typer.run(visualize_embeddings)
