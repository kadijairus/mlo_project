from __future__ import annotations
import io
from pathlib import Path
from contextlib import asynccontextmanager
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from loguru import logger
from mlo_group_project.model import BreastCancerModel
import json
import joblib  # type: ignore[import-untyped]

MODEL_PATH = Path("models/model.pth")
PROCESSED_DIR = Path("data/processed")
SCALER_PATH = PROCESSED_DIR / "scaler.joblib"
FEATURES_PATH = PROCESSED_DIR / "feature_columns.json"
LABEL_ENCODER_PATH = PROCESSED_DIR / "label_encoder.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model at {MODEL_PATH}. Train first.")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Missing scaler at {SCALER_PATH}. Run preprocessing first.")
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Missing feature columns at {FEATURES_PATH}. Run preprocessing first.")
    if not LABEL_ENCODER_PATH.exists():
        raise RuntimeError(f"Missing label encoder at {LABEL_ENCODER_PATH}. Run preprocessing first.")

    scaler = joblib.load(SCALER_PATH)
    feature_columns = json.loads(FEATURES_PATH.read_text())
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    app.state.scaler = scaler
    app.state.feature_columns = feature_columns
    app.state.label_encoder = label_encoder

    # Load model
    model = BreastCancerModel(input_shape=len(feature_columns))
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

    filename = file.filename
    if filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Drop columns
    df = df.copy()
    df.drop(columns=["id", "Unnamed: 32"], errors="ignore", inplace=True)

    has_labels = "diagnosis" in df.columns
    if has_labels:
        y_raw = df["diagnosis"]
        X_df = df.drop(columns=["diagnosis"])
    else:
        X_df = df
        y_raw = None

    # Enforce same feature columns + order as training
    feature_columns = app.state.feature_columns
    missing = set(feature_columns) - set(X_df.columns)
    extra = set(X_df.columns) - set(feature_columns)

    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {sorted(missing)}")
    if extra:
        raise HTTPException(status_code=400, detail=f"Unexpected extra columns: {sorted(extra)}")

    X_df = X_df[feature_columns]

    # Scale using training scaler
    X_scaled = app.state.scaler.transform(X_df.to_numpy())
    x = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = app.state.model(x).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    resp: dict = {
        "filename": filename,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
    }

    # Encode labels if present
    if has_labels:
        try:
            y = app.state.label_encoder.transform(y_raw).astype("float32")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not encode diagnosis labels. Expected something like 'B'/'M'. Error: {e}",
            )

    with torch.no_grad():
        logits = app.state.model(x).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    if has_labels:
        y_t = torch.tensor(y, dtype=torch.float32)
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


@app.get("/health")
def health_check():
    # Check if your model is loaded here
    return {"status": "healthy"}
