"""
scripts/shadow_compare.py
--------------------------
CI script that compares a candidate (shadow) model against the production model
on the held-out test set.

Fails with exit code 1 if:
  - The shadow model's F1 drops by more than --threshold compared to prod
  - OR the ROC-AUC drops by more than --threshold

This becomes a hard gate in the GitHub Actions pipeline.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(model, X, y, threshold=0.5) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "average_precision": float(average_precision_score(y, y_prob)),
        "f1": float(f1_score(y, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod-model",   required=True)
    parser.add_argument("--shadow-model", required=True)
    parser.add_argument("--data",         required=True)
    parser.add_argument("--labels",       required=True)
    parser.add_argument("--threshold",    type=float, default=0.01,
                        help="Max allowed F1 regression (e.g. 0.01 = 1 pp)")
    args = parser.parse_args()

    print("[shadow] Loading artifacts …")
    prod   = load(args.prod_model)
    shadow = load(args.shadow_model)
    X      = load(args.data)
    y      = load(args.labels)

    if isinstance(X, pd.DataFrame):
        X = X.values

    print("[shadow] Evaluating production model …")
    prod_metrics = evaluate(prod, X, y)
    print(f"  [PROD]   {prod_metrics}")

    print("[shadow] Evaluating shadow model …")
    shadow_metrics = evaluate(shadow, X, y)
    print(f"  [SHADOW] {shadow_metrics}")

    # ── comparison ──────────────────────────────────────────────────────────
    f1_delta   = shadow_metrics["f1"]   - prod_metrics["f1"]
    auc_delta  = shadow_metrics["roc_auc"] - prod_metrics["roc_auc"]

    result = {
        "production":  prod_metrics,
        "shadow":      shadow_metrics,
        "delta": {
            "f1":      round(f1_delta,  4),
            "roc_auc": round(auc_delta, 4),
        },
        "threshold":   args.threshold,
        "passed":      (f1_delta >= -args.threshold) and (auc_delta >= -args.threshold),
    }

    report_path = REPORTS_DIR / "shadow_comparison.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[shadow] F1 delta   : {f1_delta:+.4f}  (threshold: ±{args.threshold})")
    print(f"[shadow] AUC delta  : {auc_delta:+.4f}  (threshold: ±{args.threshold})")

    if result["passed"]:
        print("[shadow] ✓ Shadow model passed — safe to promote to production.")
        sys.exit(0)
    else:
        print("[shadow] ✗ Shadow model FAILED — regression detected. Blocking deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
