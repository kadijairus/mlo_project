from pathlib import Path
from loguru import logger
import numpy as np
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mlo_group_project.model import BreastCancerModel
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torch



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
    figure_name: str = "embeddings2.png",
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

    plt.figure(figsize=(8, 8))

    # ===== Background style (ggplot-like) =====
    ax = plt.gca()
    ax.set_facecolor("#EBEBEB")  # light gray background
    plt.grid(True, color="white", linewidth=1.0)
    ax.set_axisbelow(True)

    # ===== Class labels and colors =====
    class_map = {0: "Benign", 1: "Malignant"}
    targets_labels = np.array([class_map[int(t)] for t in targets_np])

    colors = {
        "Benign": "#2FB7B3",      # soft pink-red
        "Malignant": "#E76F6A",   # teal-blue
    }

    # ===== Train a linear classifier for straight decision boundary =====
    clf = LogisticRegression()
    clf.fit(embeddings_2d, targets_np)

    # ===== Create grid =====
    x_min, x_max = embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1
    y_min, y_max = embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # ===== Light gray background regions =====
    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.15,
        colors=["#A1A1A1", "#F0F0F0"],
    )

    # ===== Straight decision boundary (dark thick line) =====
    w = clf.coef_[0]
    b = clf.intercept_[0]
    x_vals = np.array([x_min, x_max])
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(
        x_vals,
        y_vals,
        color="#4D4D4D",
        linewidth=4,
        linestyle="-",
        zorder=5,
    )

    # ===== Scatter points =====
    for label in ["Benign", "Malignant"]:
        mask = targets_labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=35,
            color=colors[label],
            alpha=0.85,
            label=label,
        )

    # ===== Axis styling =====
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.xticks([])
    plt.yticks([])

    plt.xlabel("Component 1", fontsize=14)
    plt.ylabel("Component 2", fontsize=14)

    plt.legend(frameon=False)
    plt.title("t-SNE projection with linear decision boundary", fontsize=15)

    # ===== Save figure =====
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / figure_name
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    typer.run(visualize)
