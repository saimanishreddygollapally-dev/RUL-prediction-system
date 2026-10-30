
"""
Calibrate conformal prediction intervals on the LSTM model.

Workflow:
    1. Load trained LSTM and test data.
    2. Split test data into calibration (50%) and evaluation (50%).
    3. Calibrate conformal predictor on calibration data.
    4. Evaluate coverage on held-out evaluation data.
    5. Save the predictor and log results to MLflow.

The 50/50 split is a practical choice balancing calibration
data and evaluation data. It is not a universal standard.
"""

import joblib
import mlflow
import numpy as np
import torch

from pdm.config import (
    DEVICE,
    FEATURE_SENSORS,
    PROC,
    PROJECT_ROOT,
    RANDOM_STATE,
)
from pdm.models.conformal import ConformalPredictor
from pdm.models.lstm import RULLSTM


def main(subset: str = "FD001", alpha: float = 0.10):

    # Reproducibility for the calibration split
    np.random.seed(RANDOM_STATE)

    # --------------------------------------------------
    # 1. Load processed test data
    # --------------------------------------------------
    d = np.load(PROC / f"{subset}.npz")

    Xte = torch.tensor(
        d["Xte"],
        dtype=torch.float32
    ).to(DEVICE)

    yte = d["yte"]

    # --------------------------------------------------
    # 2. Load trained LSTM model
    # --------------------------------------------------
    model = RULLSTM(len(FEATURE_SENSORS)).to(DEVICE)

    model.load_state_dict(
        torch.load(
            PROC / "lstm.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    # --------------------------------------------------
    # 3. Generate predictions
    # --------------------------------------------------
    with torch.no_grad():
        preds = model(Xte).cpu().numpy().reshape(-1)

    # --------------------------------------------------
    # 4. Split into calibration and evaluation sets
    # --------------------------------------------------
    n = len(yte)

    idx = np.random.permutation(n)

    cal_idx = idx[:n // 2]
    eval_idx = idx[n // 2:]

    y_cal_true = yte[cal_idx]
    y_cal_pred = preds[cal_idx]

    y_eval_true = yte[eval_idx]
    y_eval_pred = preds[eval_idx]

    print(f"Calibration set: {len(cal_idx)} units")
    print(f"Evaluation set:  {len(eval_idx)} units")

    # --------------------------------------------------
    # 5. Calibrate conformal predictor
    # --------------------------------------------------
    cp = ConformalPredictor(alpha=alpha)

    q_hat = cp.calibrate(
        y_cal_true,
        y_cal_pred
    )

    print(f"\nConformal quantile (q_hat): {q_hat:.2f} cycles")

    # --------------------------------------------------
    # 6. Evaluate on held-out evaluation set
    # --------------------------------------------------
    eval_result = cp.evaluate_coverage(
        y_eval_true,
        y_eval_pred
    )

    eval_intervals = cp.predict(y_eval_pred)

    eval_covered = (
        (y_eval_true >= eval_intervals["lower"]) &
        (y_eval_true <= eval_intervals["upper"])
    )

    eval_coverage = float(eval_covered.mean())

    mean_width = float(eval_intervals["width"].mean())

    print(f"\n--- Held-out Evaluation Results (alpha={alpha}) ---")
    print(f"Target coverage:       {eval_result['target_coverage']:.0%}")
    print(f"Empirical coverage:    {eval_coverage:.0%}")
    print(f"Coverage gap:          {eval_result['coverage_gap']:+.1%}")
    print(f"Mean interval width:   {mean_width:.2f} cycles")
    print(f"Evaluation units:      {len(y_eval_true)}")

    # --------------------------------------------------
    # 7. Save calibrated predictor for API
    # --------------------------------------------------
    conformal_path = PROC / "conformal.joblib"

    joblib.dump(cp, conformal_path)

    print(f"\nSaved conformal predictor to {conformal_path}")

    # --------------------------------------------------
    # 8. Log results to MLflow
    # --------------------------------------------------
    mlflow_db_path = PROJECT_ROOT / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db_path}"
    )

    mlflow.set_experiment("rul-cmapss")

    with mlflow.start_run(run_name="conformal-calibration"):

        mlflow.log_params({
            "model": "LSTM",
            "subset": subset,
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "n_calibration": len(cal_idx),
            "n_evaluation": len(eval_idx),
            "random_state": RANDOM_STATE,
        })

        mlflow.log_metrics({
            "q_hat": q_hat,
            "empirical_coverage": eval_coverage,
            "coverage_gap": eval_coverage - (1 - alpha),
            "mean_interval_width": mean_width,
        })

        mlflow.log_artifact(str(conformal_path))

    print("\nResults successfully logged to MLflow.")

    # --------------------------------------------------
    # 9. Print example predictions from evaluation set
    # --------------------------------------------------
    print("\n--- Example Predictions (first 5 evaluation units) ---")

    print(
        f"{'Unit':>6s} {'True':>8s} {'Pred':>8s} "
        f"{'Lower':>8s} {'Upper':>8s} {'Covered':>8s}"
    )

    print("-" * 60)

    for i in range(min(5, len(y_eval_true))):

        covered = "Yes" if eval_covered[i] else "No"

        print(
            f"{eval_idx[i] + 1:6d} "
            f"{y_eval_true[i]:8.1f} "
            f"{y_eval_pred[i]:8.1f} "
            f"{eval_intervals['lower'][i]:8.1f} "
            f"{eval_intervals['upper'][i]:8.1f} "
            f"{covered:>8s}"
        )


if __name__ == "__main__":
    main()
