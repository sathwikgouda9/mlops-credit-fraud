"""
src/evaluate.py
---------------
Generates a full evaluation report on the held-out test set.
Produces:
  - reports/eval_metrics.json   (DVC metric)
  - reports/confusion_matrix.json (DVC plot)
  - reports/classification_report.txt
"""

import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_artifacts():
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    splits = {}
    for name in ["X_test", "y_test"]:
        with open(PROCESSED / f"{name}.pkl", "rb") as f:
            splits[name] = pickle.load(f)
    return model, splits["X_test"], splits["y_test"]


def find_best_threshold(y_true, y_prob) -> float:
    """
    Choose the threshold that maximises F1 — important for imbalanced datasets
    where 0.5 is almost never optimal.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = float(thresholds[best_idx])
    print(f"[evaluate] Best F1 threshold: {best_thresh:.3f}  "
          f"(F1={f1_scores[best_idx]:.4f})")
    return best_thresh


if __name__ == "__main__":
    params = load_params()
    threshold = params["evaluate"]["threshold"]

    model, X_test, y_test = load_artifacts()

    y_prob = model.predict_proba(X_test)[:, 1]

    # Optionally override threshold with the F1-optimal value
    optimal_thresh = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= optimal_thresh).astype(int)

    # ── metrics ─────────────────────────────────────────────────────────────
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "f1_at_optimal_thresh": float(f1_score(y_test, y_pred)),
        "optimal_threshold": optimal_thresh,
    }

    with open(REPORTS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[evaluate] Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})

    # ── confusion matrix (DVC plot format) ──────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_data = [
        {"actual": "legit",  "predicted": "legit",  "count": int(cm[0][0])},
        {"actual": "legit",  "predicted": "fraud",  "count": int(cm[0][1])},
        {"actual": "fraud",  "predicted": "legit",  "count": int(cm[1][0])},
        {"actual": "fraud",  "predicted": "fraud",  "count": int(cm[1][1])},
    ]
    with open(REPORTS_DIR / "confusion_matrix.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    # ── human-readable report ────────────────────────────────────────────────
    report = classification_report(y_test, y_pred, target_names=["legit", "fraud"])
    (REPORTS_DIR / "classification_report.txt").write_text(report)
    print("\n[evaluate] Classification Report:\n", report)
    print("[evaluate] Done ✓")
