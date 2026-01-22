from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mlo_group_project.model import BreastCancerModel
import numpy as np


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_tsne(
    *,
    random_state: int,
    perplexity: float,
    learning_rate: str,
    iterations: int,
) -> TSNE:
    lr: float | str
    if learning_rate.strip().lower() == "auto":
        lr = "auto"
    else:
        lr = float(learning_rate)
    try:
        return TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perplexity,
            learning_rate=lr,
            max_iter=iterations,
            init="pca",
        )
    except TypeError:
        return TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perplexity,
            learning_rate=lr,
            n_iter=iterations,
            init="pca",
        )



def visualize(
    model_checkpoint: Path = Path("models/model.pth"),
    processed_dir: Path = Path("data/processed"),
    figure_name: str = "tSNE_plot_ANN.png",
    batch_size: int = 128,
    pca_components: int = 32,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str = "auto",
    tsne_iterations: int = 1_000,
    random_state: int = 42,
) -> None:
    """Visualize learned embeddings using PCA + t-SNE."""

    device = _select_device()
    logger.info("Visualizing model embeddings")
    logger.info(f"Model checkpoint: {model_checkpoint}")
    logger.info(f"Processed dir: {processed_dir}")
    logger.info(f"Using device: {device}")

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

    x_test, y_test = torch.load(test_path)

    model = BreastCancerModel(input_shape=x_test.shape[1]).to(device)
    try:
        state_dict = torch.load(model_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    embedding_model = torch.nn.Sequential(*list(model.network.children())[:-1]).to(device)
    embedding_model.eval()
    with torch.no_grad():
        embeddings = embedding_model(x_test.to(device))
    print(embeddings)

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    embeddings, targets = [], []
    with torch.inference_mode():
        for x_batch, y_batch in torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False):
            x_batch = x_batch.to(device)
            embeddings.append(embedding_model(x_batch).cpu())
            targets.append(y_batch.cpu())

    embeddings_np = torch.cat(embeddings).numpy()
    targets_np = torch.cat(targets).numpy()
    logger.info(f"Embedding shape: {embeddings_np.shape}")

    if embeddings_np.shape[1] > pca_components:
        logger.info(f"Applying PCA to {pca_components} components")
        embeddings_np = PCA(n_components=pca_components, random_state=random_state).fit_transform(embeddings_np)

    logger.info("Applying t-SNE")
    tsne = _create_tsne(
        random_state=random_state,
        perplexity=tsne_perplexity,
        learning_rate=tsne_learning_rate,
        iterations=tsne_iterations,
    )
    embeddings_2d = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(10, 10))

    class_map = {0: "Benign", 1: "Malignant"}
    targets_labels = [class_map[int(t)] for t in targets_np]
    colors = {
        "Benign": "#2FB7B3",      # soft pink-red
        "Malignant": "#E76F6A",   # teal-blue
    }

    for label in sorted(set(targets_labels)):
        mask = np.array([lbl == label for lbl in targets_labels])
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]

        plt.scatter(x, y, label=label, alpha=0.6, color=colors[label])

        # Gaussian Ellipse
        if len(x) > 1:
            mean = [np.mean(x), np.mean(y)]
            cov = np.cov(x, y)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

            width, height = 2 * np.sqrt(eigenvalues) * 2  # 放大 factor=2
            ellipse = patches.Ellipse(mean, width, height, angle=angle, edgecolor=colors[label],
                                    facecolor="none", linestyle="--", linewidth=1.5, alpha=0.8)
            plt.gca().add_patch(ellipse)

    plt.legend()
    plt.title("t-SNE Visualization of Breast Cancer Classification ANN Prediction")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / figure_name
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Visualization saved to: {output_path}")



if __name__ == "__main__":
    typer.run(visualize)
