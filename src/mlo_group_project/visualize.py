<<<<<<< HEAD
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import typer
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import MyNeuralNet


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

=======
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mlo_group_project.model import BreastCancerModel


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def visualize(
    model_checkpoint: Path = Path("models/model.pth"),
    processed_dir: Path = Path("src/mlo_group_project/data/processed"),
    figure_name: str = "embeddings.png",
    batch_size: int = 128,
    pca_components: int = 32,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: float | str = "auto",
    tsne_iterations: int = 1_000,
    random_state: int = 42,
) -> None:
    """Visualize learned embeddings using PCA + t-SNE.

    This project trains a binary classifier on the Breast Cancer Wisconsin dataset.
    We extract a 32-dim embedding from the penultimate layer and visualize it.
    """

    device = _select_device()
    print("Visualizing model embeddings...")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Processed dir: {processed_dir}")
    print(f"Using device: {device}")

    if not model_checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_checkpoint}. "
            "Train the model first or pass --model-checkpoint."
        )

    test_path = processed_dir / "test.pt"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Processed test data not found: {test_path}. "
            "Run preprocessing first or pass --processed-dir."
        )

    model = BreastCancerModel().to(device)
    try:
        state_dict = torch.load(model_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Embedding extractor: everything except the final Linear(32 -> 1).
    if not hasattr(model, "network"):
        raise AttributeError("Expected BreastCancerModel to have attribute 'network'.")
    embedding_model = torch.nn.Sequential(*list(model.network.children())[:-1]).to(device)
    embedding_model.eval()

    x_test, y_test = torch.load(test_path)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    embeddings, targets = [], []
    with torch.inference_mode():
        for x_batch, y_batch in torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False):
            x_batch = x_batch.to(device)
            embeddings.append(embedding_model(x_batch).cpu())
            targets.append(y_batch.cpu())

    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()
    print(f"Embedding shape: {embeddings.shape}")

    if embeddings.shape[1] > pca_components:
        print(f"Applying PCA to {pca_components} components...")
        embeddings = PCA(n_components=pca_components, random_state=random_state).fit_transform(embeddings)

    print("Applying t-SNE...")
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=tsne_perplexity,
        learning_rate=tsne_learning_rate,
        n_iter=tsne_iterations,
        init="pca",
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    targets_int = targets.astype(int)
    for label in sorted(set(targets_int.tolist())):
        mask = targets_int == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=str(label), alpha=0.6)
    plt.legend()
    plt.title("t-SNE visualization of BreastCancerModel embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / figure_name
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    typer.run(visualize)
>>>>>>> e4234ae (Update embedding visualization)
