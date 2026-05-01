"""
monitoring/drift_report.py
--------------------------
Uses Evidently AI to detect data drift and model performance degradation.

This is the "secret sauce" — it answers:
  "Does the data we're scoring today look like the data we trained on?"

If it doesn't → the model will silently degrade → you need to retrain.

Run:
  python monitoring/drift_report.py --reference data/processed/X_train.pkl \
                                    --current   data/new_batch.pkl \
                                    --output    reports/drift/
"""

import argparse
import json
import logging
import pickle
import smtplib
import subprocess
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import yaml
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# ── feature definitions ────────────────────────────────────────────────────
NUMERICAL_FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_of_day"]
CATEGORICAL_FEATURES = []
TARGET_COL = "Class"


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)["monitoring"]


def load_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".pkl":
        with open(p, "rb") as f:
            obj = pickle.load(f)
        return pd.DataFrame(obj) if not isinstance(obj, pd.DataFrame) else obj
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    elif p.suffix == ".parquet":
        return pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    params: dict,
) -> dict:
    """
    Generates two artefacts:
      1. drift_report.html  — interactive visual report (share with stakeholders)
      2. drift_summary.json — machine-readable summary for alerting / CI gates
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    column_mapping = ColumnMapping(
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        target=None,   # set to TARGET_COL if labels are available in current batch
    )

    # ── 1. Full drift report (HTML) ──────────────────────────────────────
    report = Report(metrics=[
        DataDriftPreset(),        # per-feature drift + dataset-level summary
        DataQualityPreset(),      # missing values, duplicates, outliers
        DatasetDriftMetric(),     # single drift flag
        DatasetMissingValuesMetric(),
    ])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    html_path = output_dir / "drift_report.html"
    report.save_html(str(html_path))
    log.info(f"Drift HTML report saved → {html_path}")

    # ── 2. Test suite (JSON) — used for CI gating ───────────────────────
    suite = TestSuite(tests=[DataDriftTestPreset()])
    suite.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    suite_path = output_dir / "test_suite.json"
    suite.save_json(str(suite_path))

    # ── 3. Extract scalar summary ────────────────────────────────────────
    result_dict = report.as_dict()
    dataset_drift = result_dict["metrics"][2]["result"]  # DatasetDriftMetric
    n_drifted = dataset_drift.get("number_of_drifted_columns", 0)
    n_total = dataset_drift.get("number_of_columns", len(NUMERICAL_FEATURES))
    drift_share = dataset_drift.get("share_of_drifted_columns", 0.0)
    dataset_drifted = dataset_drift.get("dataset_drift", False)

    summary = {
        "dataset_drifted": dataset_drifted,
        "drifted_features": n_drifted,
        "total_features": n_total,
        "drift_share": round(drift_share, 4),
        "drift_threshold": params["drift_threshold"],
        "reference_rows": len(reference),
        "current_rows": len(current),
    }

    json_path = output_dir / "drift_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Drift summary: {summary}")

    return summary


def send_alert(summary: dict, params: dict) -> None:
    """
    Sends an email alert when drift is detected.
    Configure your SMTP credentials via environment variables:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
    """
    import os
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", 587))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")

    if not user:
        log.warning("SMTP_USER not set — skipping email alert.")
        return

    msg = EmailMessage()
    msg["Subject"] = "🚨 MLOps Alert: Data Drift Detected — Credit Fraud Model"
    msg["From"] = user
    msg["To"] = params["alert_email"]
    msg.set_content(
        f"Data drift detected in production traffic.\n\n"
        f"  Drifted features : {summary['drifted_features']} / {summary['total_features']}\n"
        f"  Drift share      : {summary['drift_share']:.2%}\n"
        f"  Reference rows   : {summary['reference_rows']:,}\n"
        f"  Current rows     : {summary['current_rows']:,}\n\n"
        f"Action: Review the full HTML report and consider retraining."
    )

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, password)
        s.send_message(msg)
    log.info(f"Alert email sent to {params['alert_email']}")


def trigger_retraining() -> None:
    """
    Kicks off a DVC repro run to retrain the model with current data.
    In a real setup, this would post to a Webhook, Airflow, or SageMaker Pipeline.
    """
    log.info("Triggering automated retraining via DVC …")
    result = subprocess.run(
        ["dvc", "repro", "--force"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode == 0:
        log.info("Retraining succeeded ✓")
    else:
        log.error(f"Retraining failed:\n{result.stderr}")


# ── entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evidently AI drift detector")
    parser.add_argument(
        "--reference",
        default=str(ROOT / "data" / "processed" / "X_train.pkl"),
        help="Reference dataset (training distribution)",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Current batch to check (new production data)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "drift"),
        help="Output directory for reports",
    )
    args = parser.parse_args()

    params = load_params()

    log.info(f"Loading reference from: {args.reference}")
    log.info(f"Loading current batch from: {args.current}")

    reference_df = load_dataframe(args.reference)
    current_df = load_dataframe(args.current)

    # Align columns (drop target if present in reference)
    ref_cols = [c for c in NUMERICAL_FEATURES if c in reference_df.columns]
    cur_cols = [c for c in NUMERICAL_FEATURES if c in current_df.columns]
    common_cols = list(set(ref_cols) & set(cur_cols))

    summary = run_drift_report(
        reference=reference_df[common_cols],
        current=current_df[common_cols],
        output_dir=Path(args.output),
        params=params,
    )

    # ── decide what to do ─────────────────────────────────────────────────
    if summary["dataset_drifted"]:
        log.warning("⚠️  Dataset drift detected!")
        send_alert(summary, params)

        if params.get("retrain_on_drift"):
            trigger_retraining()
    else:
        log.info("✓ No significant drift detected.")

    # ── CI gate: exit 1 if drift exceeds threshold ─────────────────────────
    # GitHub Actions will mark the step as failed, blocking the deployment.
    if summary["drift_share"] > params["drift_threshold"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
