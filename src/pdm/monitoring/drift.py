"""
Data drift detection using Evidently.

Compares reference and current data, generates an HTML report, and
returns dataset-level and per-feature drift results.

Supports Evidently versions where Report.run() returns None and
Report.as_dict() stores results under nested metrics/results.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from pdm.config import FEATURE_SENSORS, PROC

logger = logging.getLogger(__name__)


def _arrays_to_dataframe(data: np.ndarray) -> pd.DataFrame:
    """Convert (N, sequence_length, features) arrays to 2D summaries."""
    means = pd.DataFrame(
        data.mean(axis=1),
        columns=[f"{sensor}_mean" for sensor in FEATURE_SENSORS],
    )
    stds = pd.DataFrame(
        data.std(axis=1),
        columns=[f"{sensor}_std" for sensor in FEATURE_SENSORS],
    )
    lasts = pd.DataFrame(
        data[:, -1, :],
        columns=[f"{sensor}_last" for sensor in FEATURE_SENSORS],
    )

    return pd.concat([means, stds, lasts], axis=1)


def _is_drifted(method: str, value: float, threshold: float) -> bool:
    """Interpret a statistical test result in the appropriate direction."""
    method_lower = method.lower()

    if "p_value" in method_lower or "p-value" in method_lower:
        return value < threshold

    return value > threshold


def _find_key_recursive(obj, target_key):
    """Find the first occurrence of a key in a nested dict/list."""
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]

        for value in obj.values():
            found = _find_key_recursive(value, target_key)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = _find_key_recursive(value, target_key)
            if found is not None:
                return found

    return None


def run_drift_check(
    reference_path: str | Path = None,
    current_path: str | Path = None,
    reference_data: np.ndarray = None,
    current_data: np.ndarray = None,
    save_html: bool = True,
) -> dict:
    """
    Compare reference and current data using Evidently.

    Returns a dictionary containing dataset-level and per-feature results.
    """
    # Load reference data
    if reference_data is None:
        if reference_path is None:
            reference_path = PROC / "FD001.npz"

        with np.load(reference_path) as data:
            reference_data = data["Xtr"].copy()

    # Load current data
    if current_data is None:
        if current_path is None:
            current_path = PROC / "FD001.npz"

        with np.load(current_path) as data:
            current_data = data["Xte"].copy()

    # Convert arrays to summary DataFrames
    ref_df = _arrays_to_dataframe(reference_data)
    cur_df = _arrays_to_dataframe(current_data)

    logger.info(
        "Reference shape: %s, Current shape: %s",
        ref_df.shape,
        cur_df.shape,
    )

    # Run Evidently report. In this version, report.run() returns None.
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    # Save the HTML report
    if save_html:
        PROC.mkdir(parents=True, exist_ok=True)
        report_path = PROC / "drift_report.html"
        report.save_html(str(report_path))
        logger.info("Drift report saved to %s", report_path)

    # Extract report data
    report_dict = report.as_dict()

    # Evidently 0.4.x commonly nests these values inside metric results.
    drift_by_columns = _find_key_recursive(
        report_dict, "drift_by_columns"
    )

    if not isinstance(drift_by_columns, dict):
        logger.error(
            "Could not find drift_by_columns in Evidently report. "
            "Top-level keys: %s",
            list(report_dict.keys()),
        )
        raise ValueError(
            "Unexpected Evidently report structure: "
            "'drift_by_columns' was not found. "
            "Check the installed Evidently version."
        )

    # Extract the overall dataset drift result.
    dataset_drift_value = _find_key_recursive(
        report_dict, "dataset_drift"
    )

    features = []

    for column_name, values in drift_by_columns.items():
        if not isinstance(values, dict):
            continue

        # Evidently provides these fields for individual columns.
        drift_score = values.get(
            "drift_score",
            values.get("p_value", 0.0),
        )
        method = values.get(
            "stattest_name",
            values.get("method", "unknown"),
        )
        threshold = values.get(
            "stattest_threshold",
            values.get("threshold", 0.05),
        )

        drift_detected = values.get("drift_detected")

        # Fallback for versions that don't return drift_detected.
        if drift_detected is None:
            drift_detected = _is_drifted(
                str(method),
                float(drift_score),
                float(threshold),
            )

        features.append(
            {
                "feature": values.get("column_name", column_name),
                "drift_detected": bool(drift_detected),
                "drift_score": float(drift_score),
                "method": str(method),
                "threshold": float(threshold),
            }
        )

    n_drifted = sum(
        1 for feature in features if feature["drift_detected"]
    )
    total = len(features)
    drift_ratio = n_drifted / total if total else 0.0

    # Prefer Evidently's own dataset-level decision when available.
    if dataset_drift_value is not None:
        dataset_drift = bool(dataset_drift_value)
    else:
        dataset_drift = drift_ratio > 0.5

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_drift": dataset_drift,
        "n_drifted_features": n_drifted,
        "total_features": total,
        "drift_ratio": round(drift_ratio, 3),
        "features": features,
    }

    logger.info(
        "Drift check: %s/%s features drifted, dataset_drift=%s",
        n_drifted,
        total,
        dataset_drift,
    )

    return result


def simulate_drift(
    data: np.ndarray,
    drift_fraction: float = 0.3,
    shift_magnitude: float = 2.0,
) -> np.ndarray:
    """Shift selected features to simulate distribution drift."""
    drifted = data.copy()
    n_features = data.shape[2]
    n_drift = max(1, int(n_features * drift_fraction))

    rng = np.random.RandomState(42)
    drift_indices = rng.choice(
        n_features, size=n_drift, replace=False
    )

    for idx in drift_indices:
        feature_std = data[:, :, idx].std()
        drifted[:, :, idx] += shift_magnitude * feature_std

    logger.info(
        "Simulated drift: shifted %s/%s features by %s standard deviations",
        n_drift,
        n_features,
        shift_magnitude,
    )

    return drifted


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    data_path = PROC / "FD001.npz"

    with np.load(data_path) as data:
        Xtr = data["Xtr"].copy()
        Xte = data["Xte"].copy()

    if "--simulate" in sys.argv:
        print("=== Running with SIMULATED drift ===\n")
        current = simulate_drift(Xte)
        result = run_drift_check(
            reference_data=Xtr,
            current_data=current,
        )
    else:
        print("=== Running with original test data ===\n")
        result = run_drift_check(
            reference_data=Xtr,
            current_data=Xte,
        )

    print(f"Dataset drift detected: {result['dataset_drift']}")
    print(
        f"Features drifted: "
        f"{result['n_drifted_features']}/{result['total_features']}"
    )
    print(f"Drift ratio: {result['drift_ratio']:.1%}")

    print("\nPer-feature results:")
    print(f"{'Feature':25s} {'Drifted':>8s} {'Score':>10s} {'Method'}")
    print("-" * 75)

    for feature in result["features"]:
        marker = "YES" if feature["drift_detected"] else "no"
        print(
            f"{feature['feature']:25s} "
            f"{marker:>8s} "
            f"{feature['drift_score']:10.4f}   "
            f"{feature['method']} "
            f"(threshold={feature['threshold']})"
        )
