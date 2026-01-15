from __future__ import annotations
import io
from pathlib import Path
from contextlib import asynccontextmanager
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from loguru import logger
from mlo_group_project.model import BreastCancerModel

MODEL_PATH = Path("models/model.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model at {MODEL_PATH}. Train first.")

    # We assume BCW has 30 features after dropping id/Unnamed: 32 and diagnosis
    # If we want to accept variable input shapes, we would need to handle that in a seperate preprocessing.py file
    model = BreastCancerModel(input_shape=30)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    app.state.model = model
    logger.info(f"Model loaded from {MODEL_PATH}")
    
    yield


app = FastAPI(title="Breast Cancer Inference API", lifespan=lifespan)


@app.post("/evaluate-csv")
async def evaluate_csv(file: UploadFile = File(...)) -> dict:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Uses same preprocessing as in data.py but adapted for inference TODO: deduplicate code, preprocess correctly
    df = df.copy()
    df.drop(columns=["id", "Unnamed: 32"], errors="ignore", inplace=True)

    has_labels = "diagnosis" in df.columns
    if has_labels:
        y_raw = df["diagnosis"]
        X_df = df.drop(columns=["diagnosis"])
        # assumes diagnosis is already 0/1 OR is M/B (handle both)
        if y_raw.dtype == object:
            y = y_raw.map({"B": 0, "M": 1}).astype("float32")
        else:
            y = y_raw.astype("float32")
    else:
        X_df = df
        y = None

    # Validate shape expected by model (30)
    if X_df.shape[1] != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 feature columns, got {X_df.shape[1]}.",
        )

    x = torch.tensor(X_df.to_numpy(), dtype=torch.float32)

    with torch.no_grad():
        logits = app.state.model(x).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    resp: dict = {
        "filename": file.filename,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
    }

    if has_labels:
        y_t = torch.tensor(y.to_numpy(), dtype=torch.float32)
        if y_t.numel() != preds.numel():
            raise HTTPException(status_code=400, detail="Targets shape mismatch.")

        correct = (preds == y_t).sum().item()
        total = y_t.numel()
        accuracy = correct / total if total else 0.0

        resp.update(
            {
                "has_labels": True,
                "accuracy": float(accuracy),
                "correct": int(correct),
                "total": int(total),
            }
        )
    else:
        resp.update(
            {
                "has_labels": False,
                "message": "No diagnosis column provided; returning predictions only.",
                "predicted_positive": int(preds.sum().item()),
            }
        )

    return resp