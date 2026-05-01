"""
src/preprocess.py
-----------------
Loads the raw Kaggle Credit Card Fraud CSV, applies feature engineering,
handles class imbalance via SMOTE, and persists train/test splits.

DVC tracks:  data/raw/creditcard.csv  →  data/processed/*.pkl
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DATA = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)["preprocess"]


def load_data(path: Path) -> pd.DataFrame:
    print(f"[preprocess] Loading data from {path}")
    df = pd.read_csv(path)
    print(f"[preprocess] Shape: {df.shape}  |  Fraud rate: {df['Class'].mean():.4%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Kaggle dataset's V1-V28 are already PCA-transformed.
    We add two interpretable features on top:
      - log_amount: stabilise the heavy-tailed Amount distribution
      - hour_of_day: Time is seconds since first tx; extract the hour proxy
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    df.drop(columns=["Time", "Amount"], inplace=True)  # originals redundant
    return df


def split_and_scale(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    scale: bool,
    smote_ratio: float,
) -> tuple:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns
        )
        # Persist scaler so the API can use it at inference time
        with open(PROCESSED / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("[preprocess] Scaler fitted and saved.")

    # SMOTE — only on training data to prevent data leakage
    print(f"[preprocess] Applying SMOTE (ratio={smote_ratio}) …")
    sm = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(
        f"[preprocess] After SMOTE — train: {X_train_res.shape}  "
        f"fraud_rate: {y_train_res.mean():.4%}"
    )

    return X_train_res, X_test, y_train_res, y_test


def save_splits(X_train, X_test, y_train, y_test) -> None:
    splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    for name, obj in splits.items():
        with open(PROCESSED / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    print("[preprocess] Splits saved to data/processed/")


def write_summary(df: pd.DataFrame, y_train, y_test) -> None:
    summary = {
        "total_rows": int(len(df)),
        "features": int(df.shape[1] - 1),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
    }
    with open(PROCESSED / "preprocess_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[preprocess] Summary:", summary)


# ── entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    params = load_params()
    df = load_data(RAW_DATA)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_and_scale(
        df,
        target_col=params["target_col"],
        test_size=params["test_size"],
        random_state=params["random_state"],
        scale=params["scale_features"],
        smote_ratio=params["smote_sampling_strategy"],
    )

    save_splits(X_train, X_test, y_train, y_test)
    write_summary(df, y_train, y_test)
    print("[preprocess] Done ✓")
