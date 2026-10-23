# Model Card: RUL Prediction System

## Model Details

- **Model type:** LSTM sequence-to-value regressor
- **Framework:** PyTorch
- **Model registry:** MLflow
- **Registered model:** `rul-lstm`
- **Verified Production version:** 2
- **Task:** Remaining Useful Life (RUL) prediction
- **Additional models:** XGBoost and Temporal Convolutional Network (TCN)

## Intended Use

**Primary use:** Estimate the Remaining Useful Life of turbofan engines using multivariate operational sensor time-series data.

**Intended users:** Researchers, students, reliability engineers, and maintenance-planning teams evaluating predictive-maintenance approaches.

**Out of scope:** Safety-critical control decisions and use as the sole basis for real aircraft maintenance decisions.

## Training Data

- **Dataset:** NASA C-MAPSS FD001
- **Data type:** Simulated turbofan engine run-to-failure trajectories
- **Training sequences:** 17,731
- **Test sequences:** 100
- **Sequence length:** 30 operational cycles
- **Input features:** 14 sensor features
- **Prediction target:** Remaining Useful Life, measured in cycles

## Model Performance

Recorded evaluation results from the project:

| Model | RMSE | MAE | PHM Score |
|---|---:|---:|---:|
| LSTM | 12.94 | 9.63 | 256.94 |
| XGBoost | 15.02 | 11.18 | 360.76 |
| TCN | 18.92 | 14.01 | 998.23 |

Lower values indicate better performance for these metrics. The LSTM achieved the best results among the recorded runs.

These results should be interpreted in the context of the dataset, preprocessing pipeline, and evaluation configuration used for each run.

## Uncertainty Estimation

The project uses conformal prediction to produce an interval around an RUL estimate.

| Metric | Recorded value |
|---|---:|
| Target coverage | 90% |
| Empirical coverage | 92% |
| Conformal quantile | Approximately 22.27 cycles |
| Mean interval width | Approximately 43.5 cycles |

Coverage is measured on the evaluated data and is not a guarantee of future performance under distribution shift.

## Deployment and Serving

- FastAPI provides REST endpoints for RUL prediction, fleet prediction, health checks, and alerts.
- MLflow tracks experiments and stores registered model versions.
- The serving layer loads the registered Production LSTM model, with a checkpoint fallback path.
- Streamlit provides fleet-health, model-performance, drift-monitoring, and retraining-history interfaces.

## Monitoring

Evidently is used to compare reference data with comparison data and generate a drift report.

In the recorded comparison, 36 of 42 monitored features were flagged as drifted. This result describes that particular comparison and does not establish continuous monitoring of live engine sensors.

The project includes a retraining-history interface. Automatic drift-triggered retraining, persistent logging of every run, and automated champion/challenger promotion should only be described as implemented once their execution has been verified.

## Limitations

- Trained and evaluated using simulated NASA C-MAPSS data rather than real aircraft engine telemetry.
- Results may not generalize to other operating conditions or datasets without additional evaluation and potentially retraining.
- Prediction intervals depend on the calibration procedure and its assumptions.
- Dataset drift can affect model accuracy and interval reliability.
- The system is a prototype and is not certified for operational aircraft maintenance decisions.

## Ethical and Safety Considerations

- Predictions should support, not replace, qualified human judgment.
- Overestimating RUL may delay necessary maintenance and create safety risks.
- Performance should be evaluated under realistic operating conditions before any real-world deployment.

## Reproducibility

The project uses Python, PyTorch, MLflow, FastAPI, Evidently, and Streamlit. Refer to the repository's README and dependency files for setup instructions and execution commands.

## Author and Maintenance

Update this section with the correct project maintainer(s), repository details, and the date of the latest validated model evaluation.
