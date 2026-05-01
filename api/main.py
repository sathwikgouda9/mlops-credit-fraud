"""
api/main.py
-----------
FastAPI service that wraps the trained fraud detection model.

Key features:
  - /predict        — production model inference
  - /shadow         — runs BOTH models and logs disagreements (shadow deployment)
  - /health         — liveness probe for Docker/K8s
  - /metrics        — Prometheus-compatible counters (optional)

Run locally:
  uvicorn api.main:app --reload --port 8000
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
with open(ROOT / "params.yaml") as f:
    API_CFG = yaml.safe_load(f)["api"]

app = FastAPI(
    title="Credit Fraud Detector",
    description="MLOps demo — fraud detection with shadow deployment & drift monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── model loading ──────────────────────────────────────────────────────────
def _load_model(path: str):
    p = ROOT / path
    if not p.exists():
        log.warning(f"Model not found at {p}")
        return None
    with open(p, "rb") as f:
        model = pickle.load(f)
    log.info(f"Loaded model from {p}")
    return model


prod_model = _load_model(API_CFG["model_path"])
shadow_model = _load_model(API_CFG.get("shadow_model_path", ""))
scaler_path = ROOT / "data" / "processed" / "scaler.pkl"
scaler = pickle.load(open(scaler_path, "rb")) if scaler_path.exists() else None

# ── simple in-memory prediction log (swap for a real DB in prod) ──────────
prediction_log: list[dict] = []


# ── request/response schemas ───────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    """
    The Kaggle dataset has V1–V28 (PCA features) + Amount + Time.
    Our preprocess step drops Time/Amount and adds log_amount, hour_of_day.
    Send the engineered features directly to this endpoint.
    """
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    log_amount: float = Field(..., description="log1p(Amount)")
    hour_of_day: float = Field(..., description="(Time // 3600) % 24")
    transaction_id: Optional[str] = None


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    is_fraud: bool
    fraud_probability: float
    model_version: str = "production"
    latency_ms: float


class ShadowResponse(BaseModel):
    production: PredictionResponse
    shadow: Optional[PredictionResponse]
    models_agree: Optional[bool]


# ── helpers ────────────────────────────────────────────────────────────────
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_of_day"]


def _infer(model, features: dict, threshold: float = 0.5) -> dict:
    t0 = time.perf_counter()
    X = pd.DataFrame([features])[FEATURE_COLS]
    if scaler:
        X = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)
    prob = float(model.predict_proba(X)[0, 1])
    latency = (time.perf_counter() - t0) * 1000
    return {
        "is_fraud": prob >= threshold,
        "fraud_probability": round(prob, 6),
        "latency_ms": round(latency, 2),
    }


def _log_prediction(record: dict) -> None:
    """Background task — append to in-memory log (persist to DB/S3 in prod)."""
    prediction_log.append(record)
    if len(prediction_log) % 100 == 0:
        log.info(f"Prediction log has {len(prediction_log)} entries")


# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "prod_model_loaded": prod_model is not None,
        "shadow_model_loaded": shadow_model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionFeatures, background_tasks: BackgroundTasks):
    if prod_model is None:
        raise HTTPException(status_code=503, detail="Production model not loaded")

    features = tx.model_dump(exclude={"transaction_id"})
    result = _infer(prod_model, features)

    record = {"transaction_id": tx.transaction_id, "source": "production", **result}
    if API_CFG.get("log_predictions"):
        background_tasks.add_task(_log_prediction, record)

    return PredictionResponse(
        transaction_id=tx.transaction_id,
        model_version="production",
        **result,
    )


@app.post("/shadow", response_model=ShadowResponse)
def shadow_predict(tx: TransactionFeatures, background_tasks: BackgroundTasks):
    """
    Shadow deployment endpoint:
      1. Run production model  →  return result to caller immediately
      2. Run shadow model in background  →  log disagreements
    This lets you validate a new model on live traffic with zero user impact.
    """
    if prod_model is None:
        raise HTTPException(status_code=503, detail="Production model not loaded")

    features = tx.model_dump(exclude={"transaction_id"})
    prod_result = _infer(prod_model, features)
    prod_response = PredictionResponse(
        transaction_id=tx.transaction_id, model_version="production", **prod_result
    )

    shadow_response = None
    agrees = None

    if shadow_model is not None:
        shadow_result = _infer(shadow_model, features)
        shadow_response = PredictionResponse(
            transaction_id=tx.transaction_id, model_version="shadow", **shadow_result
        )
        agrees = prod_result["is_fraud"] == shadow_result["is_fraud"]

        if not agrees:
            log.warning(
                f"Shadow disagreement | tx={tx.transaction_id} | "
                f"prod={prod_result['fraud_probability']:.4f} | "
                f"shadow={shadow_result['fraud_probability']:.4f}"
            )
            background_tasks.add_task(
                _log_prediction,
                {
                    "transaction_id": tx.transaction_id,
                    "source": "shadow_disagreement",
                    "prod_prob": prod_result["fraud_probability"],
                    "shadow_prob": shadow_result["fraud_probability"],
                },
            )

    return ShadowResponse(
        production=prod_response,
        shadow=shadow_response,
        models_agree=agrees,
    )


@app.get("/metrics")
def metrics():
    """Basic counters — wire into Prometheus/Grafana for a real dashboard."""
    total = len(prediction_log)
    fraud_count = sum(1 for p in prediction_log if p.get("is_fraud"))
    disagreements = sum(1 for p in prediction_log if p.get("source") == "shadow_disagreement")
    return {
        "total_predictions": total,
        "fraud_detected": fraud_count,
        "fraud_rate": round(fraud_count / total, 4) if total else 0,
        "shadow_disagreements": disagreements,
    }
