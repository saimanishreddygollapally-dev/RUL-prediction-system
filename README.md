# Predictive Maintenance & Remaining Useful Life (RUL) Prediction System

An end-to-end predictive maintenance system that estimates the **Remaining Useful Life (RUL)** of aircraft engines using time-series sensor data from NASA's C-MAPSS dataset. The project combines deep learning, machine learning, uncertainty estimation, experiment tracking, data drift analysis, REST API-based inference, and an interactive monitoring dashboard.

> **Project scope:** This is an educational predictive-maintenance prototype built using the simulated NASA C-MAPSS FD001 dataset. It currently uses static benchmark data rather than live industrial sensor streams.

## Overview

Unexpected equipment failure can lead to expensive maintenance, operational downtime, and safety risks. This project uses historical engine sensor readings to estimate how many operational cycles an engine may have remaining before failure.

The system includes:

- **LSTM:** Time-series deep learning model for RUL prediction.
- **TCN:** Temporal Convolutional Network for comparison.
- **XGBoost:** Tree-based machine learning baseline with engineered statistical features.
- **Conformal Prediction:** Prediction intervals to quantify uncertainty.
- **MLflow:** Experiment tracking and model registry.
- **FastAPI:** REST API for model inference and fleet predictions.
- **Evidently:** Data drift analysis.
- **Streamlit:** Interactive fleet monitoring dashboard.
- **Automated Tests:** Validation of core project functionality.

## Architecture

```text
                NASA C-MAPSS FD001
                        |
                        v
              Data Preprocessing
                        |
                        v
             30-Cycle Window Generation
                        |
             +----------+----------+
             |          |          |
             v          v          v
           LSTM        TCN       XGBoost
             |          |          |
             +----------+----------+
                        |
                        v
                 Model Evaluation
                        |
                        v
                  MLflow Registry
                        |
                        v
                Production LSTM
                        |
                        v
                 FastAPI Service
                        |
              +---------+---------+
              |                   |
              v                   v
        Single Prediction    Fleet Prediction
              |                   |
              +---------+---------+
                        |
                        v
             Conformal Prediction
                  Intervals
                        |
                        v
              Streamlit Dashboard


       Reference Data + Comparison Data
                        |
                        v
                Evidently Analysis
                        |
                        v
                 Drift Report
                        |
                        v
              Streamlit Drift Monitor
```

The FastAPI service loads the production-registered LSTM model and conformal predictor during startup. The Streamlit dashboard communicates with the API when API (live) mode is selected.

## Dataset

### NASA C-MAPSS — FD001

The project uses the FD001 subset of the Commercial Modular Aero-Propulsion System Simulation dataset.

The dataset contains simulated aircraft engine run-to-failure trajectories and sensor measurements collected over successive operational cycles.

Key characteristics:

- Multiple engine units with time-series observations.
- Operational settings and sensor measurements.
- Run-to-failure training trajectories.
- Test trajectories with corresponding RUL labels supplied separately.

### Preprocessing Pipeline

1. Load the training, test, and RUL datasets.
2. Identify engine units and order observations by operational cycle.
3. Calculate training RUL using the difference between the maximum cycle of each engine and its current cycle.
4. Clip RUL targets at 125 cycles.
5. Select 14 relevant sensor features.
6. Fit MinMaxScaler on training sensor data and apply the fitted scaler to test data.
7. Generate overlapping sequences of 30 cycles with stride 1 for training.
8. Generate one final 30-cycle sequence per test engine.
9. Use the RUL labels supplied with the test dataset for evaluation.

For test sequences shorter than 30 cycles, the implementation pads by repeating the first available row.

### Selected Sensor Features

```text
s_2, s_3, s_4, s_7, s_8, s_9, s_11,
s_12, s_13, s_14, s_15, s_17, s_20, s_21
```

### Processed Dataset Shapes

| Dataset | Shape |
|---|---|
| Training sequences | `(17731, 30, 14)` |
| Training targets | `(17731,)` |
| Test sequences | `(100, 30, 14)` |
| Test targets | `(100,)` |

Each sequence contains 30 time steps and 14 sensor features.

## Models

### 1. LSTM — Long Short-Term Memory

The LSTM network learns temporal dependencies in sequential engine sensor data.

It processes 30-cycle sensor windows and predicts the estimated remaining useful life.

The LSTM is the primary model currently registered and served through the FastAPI inference service.

### 2. Temporal Convolutional Network (TCN)

The TCN uses temporal convolutions to learn patterns in sequential sensor measurements.

It provides an alternative deep learning architecture for comparison with the LSTM.

The reported TCN results were obtained from a 10-epoch training run and should not be interpreted as a fully optimized benchmark.

### 3. XGBoost

XGBoost provides a tree-based machine learning baseline.

Instead of directly processing the complete sensor sequence, each 30-cycle window is transformed into statistical features.

#### XGBoost Feature Engineering

Each input window has the shape:

```text
(30, 14)
```

Four statistical feature groups are extracted:

| Feature Group | Number of Features |
|---|---:|
| Mean of each sensor across the window | 14 |
| Standard deviation of each sensor | 14 |
| Latest sensor value | 14 |
| Last-cycle minus first-cycle change | 14 |
| **Total** | **56** |

Resulting feature matrix shapes:

```text
Training Features: (17731, 56)
Test Features:     (100, 56)
```

#### XGBoost Configuration

```python
XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

## Model Evaluation

The models are evaluated using:

- **RMSE:** Root Mean Squared Error.
- **MAE:** Mean Absolute Error.
- **PHM Score:** Asymmetric scoring function used for RUL prediction evaluation.

### Recorded Benchmark Results

| Model | RMSE | MAE | PHM Score |
|---|---:|---:|---:|
| LSTM | 12.94 | 9.63 | 256.94 |
| XGBoost | 15.02 | 11.18 | 360.76 |
| TCN | 18.92 | 14.01 | 998.23 |

Lower values indicate better performance for all three metrics.

Among these recorded runs, the LSTM achieved the lowest RMSE, MAE, and PHM score.

> Note: These are previously recorded experimental results. The models were not necessarily trained with identical optimization budgets or compute configurations.

## Uncertainty Estimation with Conformal Prediction

A point prediction alone does not communicate the uncertainty associated with an RUL estimate.

To address this, the project implements conformal prediction for the LSTM model.

The conformal predictor uses calibration residuals to estimate a quantile and construct prediction intervals around the model's point predictions.

### Latest Calibration Results

| Metric | Result |
|---|---:|
| Target Coverage | 90% |
| Empirical Coverage | 92% |
| Conformal Quantile (`q_hat`) | 22.21 cycles |
| Mean Interval Width | 43.37 cycles |
| Calibration Units | 50 |
| Evaluation Units | 50 |

### Prediction Interval

The prediction interval is calculated using:

```text
Lower Bound = max(0, Predicted RUL - q_hat)

Upper Bound = Predicted RUL + q_hat
```

Where:

```text
q_hat = 22.21 cycles
```

The lower bound is clipped at zero to prevent negative RUL estimates.

### Evaluation Methodology

The official FD001 test set was divided into two subsets:

- 50 units for conformal calibration.
- 50 units for held-out evaluation.

The evaluation subset was not used to calculate the calibration quantile.

The observed empirical coverage was 92%, compared with the target coverage of 90%.

**Important limitation:** Both subsets come from the official FD001 test set. Consequently, this is a held-out evaluation relative to calibration, but not an independent external validation. The reported coverage should not be interpreted as proof of guaranteed coverage under future distribution shifts.

Conformal calibration is currently implemented for the LSTM model only. Other models would require their own calibration residuals and fitted quantiles.

## MLflow — Experiment Tracking & Model Registry

MLflow is used to track model experiments and manage registered models.

Implemented capabilities include:

- Experiment tracking.
- Logging model evaluation metrics.
- Logging model artifacts.
- Model registration.
- Model versioning.
- Production-stage model selection.
- Loading the registered LSTM model for inference.

The registered LSTM model's Production version was verified during development:

```text
Model Name:    rul-lstm
Model Version: 2
Model Stage:   Production
```

The FastAPI service loads the registered production model during application startup.

> MLflow model stages are deprecated in newer MLflow versions. The current implementation still uses the Production stage and can be migrated to model aliases in a future update.

## FastAPI Inference Service

FastAPI exposes the prediction functionality through REST endpoints.

The model and conformal predictor are loaded once during application startup and reused for subsequent requests.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Checks service health and loaded model information |
| POST | `/predict` | Predicts RUL for a single engine sequence |
| GET | `/predict/fleet` | Generates predictions for all test engine units |
| GET | `/alerts` | Retrieves fleet alerts based on an RUL threshold |

### Single-Unit Prediction

The `/predict` endpoint accepts:

- 30 time steps.
- 14 already-scaled sensor values per time step.

It returns:

- Predicted RUL.
- Lower prediction interval.
- Upper prediction interval.
- Engine health status.

### Fleet Prediction

The `/predict/fleet` endpoint:

1. Loads the processed FD001 test dataset.
2. Runs batch inference using the production LSTM.
3. Applies the loaded conformal predictor.
4. Calculates fleet-level health statistics.
5. Returns individual unit predictions and uncertainty intervals.

### Health Check

The `/health` endpoint reports:

```json
{
    "status": "healthy",
    "model_name": "rul-lstm",
    "model_version": "2",
    "model_stage": "Production"
}
```

### Interactive API Documentation

FastAPI automatically provides Swagger documentation.

```text
http://localhost:8000/docs
```

Additional endpoints:

```text
http://localhost:8000/health
http://localhost:8000/redoc
```

### API Integration Verification

During development, the following were verified:

- FastAPI started successfully.
- The health endpoint returned `healthy`.
- The production model was reported as `rul-lstm`, version 2.
- Streamlit successfully requested `/predict/fleet`.
- FastAPI returned HTTP `200 OK` responses.

The fleet endpoint currently reads the static processed FD001 test dataset. It does not ingest live industrial sensor measurements.

## Streamlit Dashboard

The Streamlit dashboard provides an interactive interface for monitoring fleet predictions and model information.

### Data Source Modes

The dashboard supports two modes.

**1. Local (from files)**

- Loads the processed FD001 dataset.
- Runs local LSTM inference.
- Applies the local conformal predictor when the calibration artifact exists.

**2. API (live)**

- Sends requests to the FastAPI backend.
- Retrieves fleet predictions through `/predict/fleet`.
- Displays predictions and conformal intervals returned by the API.

Here, "live" refers to a live API request, not live industrial sensor ingestion.

### Dashboard Sections

#### Fleet Health

Displays:

- Total engine units.
- Critical engine count.
- Warning engine count.
- Healthy engine count.
- Average fleet RUL.
- Fleet RUL distribution.
- Maintenance priority queue.
- Unit-level predictions with uncertainty intervals.

#### Model Performance

Displays:

- LSTM performance.
- TCN performance.
- XGBoost performance.
- RMSE comparison.
- MAE and PHM scores.
- Conformal calibration information.

#### Drift Monitor

Displays the latest Evidently drift report and associated drift statistics.

#### Retrain History

Displays available retraining history, including recorded pipeline runs and model evaluation information.

> Development note: The current dashboard code falls back to local inference when the API request fails. Therefore, the presence of predictions on the dashboard alone does not prove API connectivity. Check the Streamlit warning and FastAPI access logs when verifying API mode.

## Drift Monitoring with Evidently

The project uses Evidently to compare reference data against comparison data and identify changes in feature distributions.

The previously recorded comparison reported:

```text
Drifted Features: 36
Total Features:   42
Drift Ratio:      85.7%
```

The workflow is:

```text
Reference Dataset
        |
        v
Comparison Dataset
        |
        v
Evidently Drift Analysis
        |
        v
Drift Report
        |
        v
Streamlit Drift Monitor
```

This demonstrates an offline data-drift analysis workflow.

It does not represent continuous monitoring of real engine sensor streams. Continuous monitoring would require a sensor-data ingestion pipeline, persistent storage, and scheduled or streaming drift evaluations.

## Automated Retraining Pipeline

The project includes a retraining pipeline designed to support:

- Drift-based retraining workflows.
- Candidate model training.
- Model evaluation.
- Comparison against the existing champion model.
- Model promotion decisions.
- Retraining history logging.

The Streamlit dashboard provides a Retrain History section for displaying available pipeline records.

The existence of the pipeline and dashboard does not, by itself, establish continuous autonomous retraining in a deployed production environment.

## Running the Project Locally

### Prerequisites

- Python compatible with the project dependencies.
- Git.
- Virtual environment.
- NASA C-MAPSS FD001 dataset.
- Required processed data and model artifacts.

### 1. Clone the Repository

```bash
git clone https://github.com/saimanishreddygollapally-dev/RUL-prediction-system.git

cd RUL-prediction-system
```

### 2. Create and Activate a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Python Path

From the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

### 5. Prepare the Dataset

Run the dataset-building pipeline:

```powershell
python -m pdm.data.build_dataset
```

Ensure the required FD001 source files are available in the locations expected by the project configuration.

### 6. Start the FastAPI Service

Open a PowerShell terminal in the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"

.\.venv\Scripts\python.exe -m uvicorn pdm.serving.api:app --host 127.0.0.1 --port 8000
```

Wait for the application startup to complete.

Useful URLs:

- Swagger documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

### 7. Start the Streamlit Dashboard

Open a second PowerShell terminal in the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"

.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

To use the FastAPI backend:

1. Select **API (live)** in the dashboard sidebar.
2. Set the endpoint to `http://localhost:8000`.
3. Keep the FastAPI server running.

### 8. Run Conformal Calibration

To generate the conformal calibration artifact:

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m pdm.models.calibrate
```

The generated artifact is saved to:

```text
data/processed/conformal.joblib
```

Restart FastAPI after calibration so the API loads the updated artifact into memory.

### 9. Run Tests

```powershell
$env:PYTHONPATH = "$PWD\src"

.\.venv\Scripts\python.exe -m pytest
```

The project previously reported 23 passing tests during development. Run the test suite against the current checkout before treating that count as a current result.

## Project Structure

Simplified high-level project structure:

```text
predictive-maintenance-rul-main/
|
├── app/
|   └── streamlit_app.py
|
├── src/
|   └── pdm/
|       ├── data/
|       ├── evaluate/
|       ├── features/
|       ├── models/
|       ├── monitoring/
|       ├── pipeline/
|       └── serving/
|           └── api.py
|
├── data/
|   └── processed/
|
├── tests/
|
├── requirements.txt
├── README.md
└── .gitignore
```

This is a simplified overview. Refer to the repository for the complete module and file structure.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | NumPy, Pandas |
| Experiment Tracking | MLflow |
| Model Serving | FastAPI, Pydantic |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Drift Monitoring | Evidently |
| Serialization | Joblib |
| Testing | Pytest |

## Key Engineering Highlights

- Time-series preprocessing and fixed-length sequence generation.
- RUL prediction using LSTM and TCN architectures.
- Statistical feature engineering for XGBoost.
- Model comparison using RMSE, MAE, and PHM score.
- Conformal prediction for uncertainty estimation.
- Model versioning and production model loading through MLflow.
- REST-based inference using FastAPI.
- Fleet-level predictions and maintenance priority identification.
- Interactive monitoring dashboard using Streamlit.
- Dataset drift analysis using Evidently.
- Modular Python implementation and automated testing.

## Limitations & Future Improvements

Current limitations:

- Experiments use simulated NASA C-MAPSS FD001 data rather than real operational engine telemetry.
- FD001 represents one operating condition and one fault mode.
- The fleet API endpoint reads a static processed test dataset.
- Conformal calibration is currently implemented for LSTM only.
- Calibration and evaluation subsets are both derived from the official test set.
- The reported TCN result comes from a 10-epoch run.
- Drift analysis is an offline comparison rather than continuous production monitoring.
- The current API uses MLflow's deprecated Production-stage model selection.

Potential improvements:

- Extend experiments to FD002, FD003, and FD004.
- Integrate a live sensor simulator or real sensor-data source.
- Calibrate and evaluate uncertainty intervals for TCN and XGBoost.
- Use an independent calibration and test dataset.
- Persist retraining records with timestamps, metrics, and model versions.
- Implement stronger candidate-model validation before promotion.
- Migrate MLflow model-stage selection to model aliases.
- Add authentication and access control for deployed services.
- Deploy the API and dashboard in a production environment.
- Monitor model performance and prediction quality over time.

## Disclaimer

This project is an educational predictive-maintenance prototype built using simulated engine data.

Its predictions are not certified for real aircraft maintenance or operational safety decisions.

## Author

**Sai Manish Reddy Gollapally**

GitHub: https://github.com/saimanishreddygollapally-dev

Repository: https://github.com/saimanishreddygollapally-dev/RUL-prediction-system