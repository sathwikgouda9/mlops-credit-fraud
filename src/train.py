"""
src/train.py
------------
Trains a Random Forest classifier on the preprocessed fraud data.
Every hyperparameter, metric, and the final model artifact are
logged to MLflow so you can compare runs in the UI.

Run:   python src/train.py
UI:    mlflow ui  (then open http://localhost:5000)
"""

import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_splits():
    splits = {}
    for name in ["X_train", "y_train", "X_test", "y_test"]:
        with open(PROCESSED / f"{name}.pkl", "rb") as f:
            splits[name] = pickle.load(f)
    return splits["X_train"], splits["y_train"], splits["X_test"], splits["y_test"]


def build_model(p: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        class_weight=p["class_weight"],
        random_state=p["random_state"],
        n_jobs=-1,
    )


def compute_metrics(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_params = load_params()
    tp = all_params["train"]
    ep = all_params["evaluate"]

    # ── MLflow experiment setup ──────────────────────────────────────────
    mlflow.set_tracking_uri("file:///C:/ML Projects/mlops-credit-fraud/mlruns")   # or set MLFLOW_TRACKING_URI env var
    mlflow.set_experiment(tp["mlflow_experiment"])

    with mlflow.start_run(run_name=tp["mlflow_run_name"]) as run:
        print(f"[train] MLflow run ID: {run.info.run_id}")

        # ── log every param so the run is fully reproducible ────────────
        mlflow.log_params({
            "model_type": tp["model_type"],
            "n_estimators": tp["n_estimators"],
            "max_depth": tp["max_depth"],
            "min_samples_split": tp["min_samples_split"],
            "class_weight": tp["class_weight"],
            "random_state": tp["random_state"],
            "smote_ratio": all_params["preprocess"]["smote_sampling_strategy"],
            "test_size": all_params["preprocess"]["test_size"],
            "eval_threshold": ep["threshold"],
        })

        # ── load data & train ────────────────────────────────────────────
        X_train, y_train, X_test, y_test = load_splits()
        print(f"[train] Training on {X_train.shape[0]:,} samples …")

        model = build_model(tp)
        model.fit(X_train, y_train)
        print("[train] Training complete ✓")

        # ── evaluate & log metrics ───────────────────────────────────────
        metrics = compute_metrics(model, X_test, y_test, ep["threshold"])
        mlflow.log_metrics(metrics)
        print("[train] Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})

        # ── log model artifact ───────────────────────────────────────────
        # Signature auto-infers input/output schema — useful for the model registry
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="CreditFraudDetector",
        )

        # ── persist locally for DVC tracking ────────────────────────────
        model_path = MODELS_DIR / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        metadata = {
            "run_id": run.info.run_id,
            "model_type": tp["model_type"],
            "mlflow_experiment": tp["mlflow_experiment"],
            **metrics,
        }
        with open(MODELS_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # ── DVC-compatible metrics file ──────────────────────────────────
        with open(REPORTS_DIR / "train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[train] Model saved → {model_path}")
        print(f"[train] View run:  mlflow ui  →  Experiment '{tp['mlflow_experiment']}'")
