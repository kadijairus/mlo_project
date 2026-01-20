# main.py
from __future__ import annotations

import io
import json
import os
import tempfile

import pandas as pd
import torch
from flask import Request, jsonify

import joblib
from google.cloud import storage

# Import your model class
from mlo_group_project.model import BreastCancerModel


BUCKET = os.environ.get("MODEL_BUCKET", "test-model-bucket-1996")
PREFIX = os.environ.get("ARTIFACT_PREFIX", "artifacts")

# Local cache paths in the function instance
CACHE_DIR = "/tmp/artifacts"
MODEL_PATH = f"{CACHE_DIR}/model.pth"
SCALER_PATH = f"{CACHE_DIR}/scaler.joblib"
FEATURES_PATH = f"{CACHE_DIR}/feature_columns.json"
LABEL_ENCODER_PATH = f"{CACHE_DIR}/label_encoder.joblib"

_loaded = False
_model = None
_scaler = None
_feature_columns = None
_label_encoder = None


def _download_if_needed():
    global _loaded, _model, _scaler, _feature_columns, _label_encoder

    if _loaded:
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    def dl(blob_name: str, dst: str):
        blob = bucket.blob(f"{PREFIX}/{blob_name}")
        if not os.path.exists(dst):
            blob.download_to_filename(dst)

    dl("model.pth", MODEL_PATH)
    dl("scaler.joblib", SCALER_PATH)
    dl("feature_columns.json", FEATURES_PATH)
    dl("label_encoder.joblib", LABEL_ENCODER_PATH)

    _scaler = joblib.load(SCALER_PATH)
    _feature_columns = json.loads(open(FEATURES_PATH).read())
    _label_encoder = joblib.load(LABEL_ENCODER_PATH)

    _model = BreastCancerModel(input_shape=len(_feature_columns))
    state = torch.load(MODEL_PATH, map_location="cpu")
    _model.load_state_dict(state)
    _model.eval()

    _loaded = True


def evaluate_csv(request: Request):
    """
    HTTP Cloud Function entrypoint.
    Expects multipart/form-data with a file field named 'file' (same as your Streamlit/FastAPI).
    """
    try:
        _download_if_needed()

        if "file" not in request.files:
            return jsonify({"detail": "Missing multipart file field 'file'"}), 400

        f = request.files["file"]
        filename = f.filename or "uploaded.csv"
        if not filename.lower().endswith(".csv"):
            return jsonify({"detail": "Please upload a .csv file."}), 400

        raw = f.read()
        df = pd.read_csv(io.BytesIO(raw))

        df = df.copy()
        df.drop(columns=["id", "Unnamed: 32"], errors="ignore", inplace=True)

        has_labels = "diagnosis" in df.columns
        if has_labels:
            y_raw = df["diagnosis"]
            X_df = df.drop(columns=["diagnosis"])
        else:
            X_df = df
            y_raw = None

        missing = set(_feature_columns) - set(X_df.columns)
        extra = set(X_df.columns) - set(_feature_columns)
        if missing:
            return jsonify({"detail": f"Missing columns: {sorted(missing)}"}), 400
        if extra:
            return jsonify({"detail": f"Unexpected extra columns: {sorted(extra)}"}), 400

        X_df = X_df[_feature_columns]
        X_scaled = _scaler.transform(X_df.to_numpy())
        x = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = _model(x).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

        resp = {
            "filename": filename,
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
        }

        if has_labels:
            try:
                y = _label_encoder.transform(y_raw).astype("float32")
            except Exception as e:
                return (
                    jsonify(
                        {
                            "detail": "Could not encode diagnosis labels. Expected something like 'B'/'M'.",
                            "error": str(e),
                        }
                    ),
                    400,
                )

            y_t = torch.tensor(y, dtype=torch.float32)
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

        return jsonify(resp), 200

    except Exception as e:
        return jsonify({"detail": "Internal error", "error": str(e)}), 500