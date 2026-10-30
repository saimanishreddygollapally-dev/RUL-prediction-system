
"""
Conformal prediction for calibrated RUL uncertainty intervals.

Features:
- Split conformal prediction for regression.
- Finite-sample corrected quantile calculation.
- Model-agnostic: applicable to LSTM, TCN, XGBoost, etc.
- Nonnegative RUL prediction intervals.
- Coverage and interval-width evaluation.

Coverage guarantee:
Standard split conformal prediction provides marginal coverage
under exchangeability between calibration and future observations.
"""

import numpy as np


class ConformalPredictor:
    """
    Split conformal predictor for regression.

    Usage:
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(y_cal_true, y_cal_pred)

        intervals = cp.predict(y_new_pred)

        # Access:
        # intervals["lower"]
        # intervals["upper"]
        # intervals["width"]
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Miscoverage rate.
                   0.10 = 90% target coverage.
                   0.05 = 95% target coverage.
        """
        if not 0 < alpha < 1:
            raise ValueError(
                f"alpha must be between 0 and 1, got {alpha}"
            )

        self.alpha = alpha
        self.q_hat = None
        self.scores = None

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate the conformal quantile from calibration residuals.

        Args:
            y_true: Actual RUL values from calibration data.
            y_pred: Model predictions for calibration data.

        Returns:
            q_hat: Finite-sample corrected residual quantile.
        """
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, "
                f"y_pred={len(y_pred)}"
            )

        if len(y_true) == 0:
            raise ValueError("Calibration data cannot be empty")

        if not np.all(np.isfinite(y_true)):
            raise ValueError(
                "Calibration targets contain NaN or infinite values"
            )

        if not np.all(np.isfinite(y_pred)):
            raise ValueError(
                "Calibration predictions contain NaN or infinite values"
            )

        # Absolute residuals (nonconformity scores)
        self.scores = np.abs(y_true - y_pred)

        # Finite-sample corrected order statistic
        n = len(self.scores)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))

        if k > n:
            self.q_hat = float("inf")
        else:
            self.q_hat = float(np.sort(self.scores)[k - 1])

        return self.q_hat

    def predict(self, y_pred: np.ndarray) -> dict:
        """
        Generate nonnegative RUL prediction intervals.

        Args:
            y_pred: Point predictions from the trained model.

        Returns:
            Dictionary containing:
                point: Original model predictions.
                lower: Lower prediction bounds, clipped at zero.
                upper: Upper prediction bounds.
                width: Width of the clipped intervals.
                q_hat: Calibration quantile.
        """
        if self.q_hat is None:
            raise RuntimeError(
                "Call calibrate() before predict()"
            )

        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        if not np.all(np.isfinite(y_pred)):
            raise ValueError(
                "Predictions contain NaN or infinite values"
            )

        # Construct symmetric conformal intervals
        raw_lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat

        # RUL cannot be negative
        lower = np.maximum(raw_lower, 0.0)

        # Width after clipping the lower bound
        width = upper - lower

        return {
            "point": y_pred,
            "lower": lower,
            "upper": upper,
            "width": width,
            "q_hat": self.q_hat,
        }

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict:
        """
        Evaluate empirical coverage on held-out evaluation data.

        Coverage is calculated using the nonnegative-clipped
        prediction intervals returned by predict().

        Note:
            The standard split-conformal marginal coverage
            guarantee applies to the original symmetric intervals.
            Clipping the lower bound at zero changes those intervals.
        """
        if self.q_hat is None:
            raise RuntimeError(
                "Call calibrate() before evaluate_coverage()"
            )

        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, "
                f"y_pred={len(y_pred)}"
            )

        if len(y_true) == 0:
            raise ValueError("Evaluation data cannot be empty")

        if not np.all(np.isfinite(y_true)):
            raise ValueError(
                "Evaluation targets contain NaN or infinite values"
            )

        intervals = self.predict(y_pred)

        covered = (
            (y_true >= intervals["lower"]) &
            (y_true <= intervals["upper"])
        )

        coverage = float(covered.mean())
        mean_width = float(intervals["width"].mean())

        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": coverage,
            "coverage_gap": coverage - (1 - self.alpha),
            "mean_interval_width": mean_width,
            "median_interval_width": float(
                np.median(intervals["width"])
            ),
            "q_hat": self.q_hat,
            "n_test": len(y_true),
        }
