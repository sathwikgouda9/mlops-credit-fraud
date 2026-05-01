"""
tests/test_preprocess.py
------------------------
Unit tests for the preprocessing module.
GitHub Actions runs these on every push.

Run locally:   pytest tests/ -v --cov=src
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Import the functions we want to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import engineer_features, split_and_scale


# ── fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture
def raw_df():
    """Minimal synthetic fraud dataset (mimics Kaggle structure)."""
    np.random.seed(42)
    n = 1000
    n_fraud = 50
    df = pd.DataFrame({
        **{f"V{i}": np.random.randn(n) for i in range(1, 29)},
        "Amount": np.abs(np.random.exponential(80, n)),
        "Time": np.arange(n) * 10.0,
        "Class": [1] * n_fraud + [0] * (n - n_fraud),
    })
    return df


@pytest.fixture
def engineered_df(raw_df):
    return engineer_features(raw_df)


# ── engineer_features ──────────────────────────────────────────────────────
class TestEngineerFeatures:
    def test_drops_time_and_amount(self, raw_df):
        result = engineer_features(raw_df)
        assert "Time" not in result.columns
        assert "Amount" not in result.columns

    def test_adds_log_amount(self, raw_df):
        result = engineer_features(raw_df)
        assert "log_amount" in result.columns

    def test_adds_hour_of_day(self, raw_df):
        result = engineer_features(raw_df)
        assert "hour_of_day" in result.columns

    def test_log_amount_non_negative(self, raw_df):
        result = engineer_features(raw_df)
        assert (result["log_amount"] >= 0).all()

    def test_hour_of_day_range(self, raw_df):
        result = engineer_features(raw_df)
        assert result["hour_of_day"].between(0, 23).all()

    def test_does_not_mutate_input(self, raw_df):
        original_cols = set(raw_df.columns)
        engineer_features(raw_df)
        assert set(raw_df.columns) == original_cols

    def test_row_count_preserved(self, raw_df):
        result = engineer_features(raw_df)
        assert len(result) == len(raw_df)

    def test_log_amount_formula(self, raw_df):
        result = engineer_features(raw_df)
        expected = np.log1p(raw_df["Amount"])
        np.testing.assert_allclose(result["log_amount"].values, expected.values)


# ── split_and_scale ────────────────────────────────────────────────────────
class TestSplitAndScale:
    def test_split_sizes(self, engineered_df):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.preprocess.PROCESSED", Path(tmp)):
                X_train, X_test, y_train, y_test = split_and_scale(
                    engineered_df,
                    target_col="Class",
                    test_size=0.2,
                    random_state=42,
                    scale=False,
                    smote_ratio=0.1,
                )
        # SMOTE enlarges training set, so just check test size roughly
        total = len(y_train) + len(y_test)
        assert abs(len(y_test) / total - 0.2) < 0.05

    def test_target_removed_from_features(self, engineered_df):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.preprocess.PROCESSED", Path(tmp)):
                X_train, X_test, y_train, y_test = split_and_scale(
                    engineered_df, "Class", 0.2, 42, False, 0.1
                )
        assert "Class" not in X_train.columns
        assert "Class" not in X_test.columns

    def test_scaler_saved(self, engineered_df):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch("src.preprocess.PROCESSED", tmp_path):
                split_and_scale(engineered_df, "Class", 0.2, 42, True, 0.1)
            assert (tmp_path / "scaler.pkl").exists()

    def test_no_label_leakage_via_smote(self, engineered_df):
        """SMOTE must only see training data — test set stays untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.preprocess.PROCESSED", Path(tmp)):
                X_train, X_test, y_train, y_test = split_and_scale(
                    engineered_df, "Class", 0.2, 42, False, 0.1
                )
        # Test fraud rate should match the original (no synthetic samples added)
        original_fraud_rate = engineered_df["Class"].mean()
        test_fraud_rate = y_test.mean()
        assert abs(test_fraud_rate - original_fraud_rate) < 0.05

    def test_smote_increases_training_fraud_rate(self, engineered_df):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.preprocess.PROCESSED", Path(tmp)):
                _, _, y_train, _ = split_and_scale(
                    engineered_df, "Class", 0.2, 42, False, smote_ratio=0.5
                )
        # After SMOTE with ratio=0.5, fraud should be ~33% of training set
        assert y_train.mean() > 0.15


# ── data quality checks ────────────────────────────────────────────────────
class TestDataQuality:
    def test_no_nulls_after_engineering(self, raw_df):
        result = engineer_features(raw_df)
        assert result.isnull().sum().sum() == 0

    def test_feature_dtypes_numeric(self, engineered_df):
        feature_cols = [c for c in engineered_df.columns if c != "Class"]
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(engineered_df[col]), \
                f"{col} is not numeric"

    def test_class_binary(self, raw_df):
        assert set(raw_df["Class"].unique()).issubset({0, 1})
