from pathlib import Path
import matplotlib.pyplot as plt
import torch
from loguru import logger
import typer
import numpy as np

from mlo_group_project.model import BreastCancerModel


def _select_device() -> torch.device:
    """Automatically select the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def embedding_sensitivity_volcano(
    model_checkpoint: Path = Path("models/model.pth"),
    processed_dir: Path = Path("data/processed"),
    figure_name: str = "embedding_volcano_top10.png",
    batch_size: int = 128,
    top_k: int | None = 200,
    annotate_top10: bool = True,
) -> None:
    """
    Embedding Sensitivity Volcano Plot

    Computes gradient-based sensitivity of embeddings w.r.t. input features
    and visualizes them in a volcano plot.

    Args:
        model_checkpoint: Path to saved model checkpoint.
        processed_dir: Path to preprocessed dataset directory.
        figure_name: Name of the output figure.
        batch_size: Batch size for gradient computation.
        top_k: Only display the top_k most sensitive points.
        annotate_top10: If True, label the top-10 most sensitive points.
    """
    device = _select_device()
    logger.info(f"Device: {device}")

    # ------------------------------
    # 1️⃣ Load test data
    # ------------------------------
    test_path = processed_dir / "test.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test data not found: {test_path}")
    x_test, y_test = torch.load(test_path)

    # ------------------------------
    # 2️⃣ Load model
    # ------------------------------
    model = BreastCancerModel(input_shape=x_test.shape[1]).to(device)
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------
    # 3️⃣ Extract embedding model (remove final layer)
    # ------------------------------
    embedding_model = torch.nn.Sequential(*list(model.network.children())[:-1]).to(device)
    embedding_model.eval()

    # ------------------------------
    # 4️⃣ Compute gradient sensitivity matrix
    # ------------------------------
    feature_dim = x_test.shape[1]
    with torch.no_grad():
        emb_sample = embedding_model(x_test[:1].to(device))
    embedding_dim = emb_sample.shape[1]

    sens_matrix = torch.zeros(embedding_dim, feature_dim, device=device)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("Computing sensitivity matrix...")
    for x_batch, _ in loader:
        # Clone and enable gradient
        x_batch = x_batch.clone().detach().to(device).requires_grad_(True)
        emb_batch = embedding_model(x_batch)
        for d in range(embedding_dim):
            x_batch.grad = None
            # Compute gradient of embedding dim d w.r.t. input
            emb_batch[:, d].mean().backward(retain_graph=True)
            sens_matrix[d] += x_batch.grad.abs().sum(dim=0)

    sens_matrix /= len(x_test)  # Average over samples

    # # ------------------------------
    # # 5️⃣ Select top-k points
    # # ------------------------------
    # if top_k is not None:
    #     flat = sens_matrix.flatten()
    #     thresh = torch.topk(flat, top_k).values.min()
    #     mask = sens_matrix >= thresh
    # else:
    #     mask = torch.ones_like(sens_matrix, dtype=torch.bool)

    # ------------------------------
    # 6️⃣ Volcano plot
    # ------------------------------
    feature_sens = sens_matrix.mean(dim=0).cpu().numpy()  # average over embeddings
    feature_idx = np.arange(feature_sens.shape[0])

    plt.figure(figsize=(12,6))
    plt.scatter(feature_idx, feature_sens, s=100, c=feature_sens, cmap="viridis", alpha=0.8)
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Sensitivity")
    plt.title("Feature-level Embedding Sensitivity Volcano")
    plt.colorbar(label="Mean Sensitivity")
    # Highlight top-10
    top10_idx = np.argsort(feature_sens)[-10:]
    for i in top10_idx:
        plt.scatter(i, feature_sens[i], s=200, facecolors="none", edgecolors="red", linewidths=2)
        plt.text(i, feature_sens[i], f"{feature_sens[i]:.2f}", ha='center', va='bottom', color='red')
    plt.show()

    # ------------------------------
    # 7️⃣ Annotate top-10 most sensitive points
    # ------------------------------
    if annotate_top10:
        flat_vals = sens_matrix.flatten()
        top10_idx = torch.topk(flat_vals, 10).indices
        for idx in top10_idx:
            emb_idx = int((idx // feature_dim).cpu())
            feat_idx = int((idx % feature_dim).cpu())
            val = sens_matrix[emb_idx, feat_idx].item()
            plt.scatter(feat_idx, emb_idx, s=300, facecolors='none', edgecolors='red', linewidths=2)
            plt.text(feat_idx, emb_idx, f"{val:.2f}", color='red', fontsize=9, ha='center', va='bottom')


    # ------------------------------
    # 8️⃣ Save figure
    # ------------------------------
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / figure_name
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Volcano plot saved to: {output_path}")


if __name__ == "__main__":
    typer.run(embedding_sensitivity_volcano)
