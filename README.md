# Predictive Maintenance & Remaining Useful Life (RUL) Prediction

An end-to-end predictive maintenance system that estimates the **Remaining Useful Life (RUL)** of aircraft engines using time-series sensor data from NASA's C-MAPSS dataset. The project combines deep learning, machine learning, uncertainty estimation, model tracking, drift monitoring, and API-based deployment in an interactive dashboard.

## Overview

Unexpected equipment failure can lead to expensive maintenance, operational downtime, and safety risks. This project uses historical engine sensor readings to estimate how many operational cycles an engine may have remaining before failure.

The system includes:
- **LSTM:** Time-series deep learning model for RUL prediction.
- **TCN:** Temporal Convolutional Network for comparison.
- **XGBoost:** Machine learning baseline.
- **Conformal prediction:** Prediction intervals to represent uncertainty.
- **MLflow:** Experiment tracking and model registry.
- **FastAPI:** REST API for model inference and fleet predictions.
- **Evidently:** Data drift analysis.
- **Streamlit:** Interactive monitoring dashboard.
- **Automated tests:** Validation of core project functionality.

## Architecture

```text
NASA C-MAPSS Dataset
        |
        v
Data Preprocessing & Feature Engineering
        |
        v
Time-Series Window Generation
        |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
      LSTM                TCN              XGBoost
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                 Model Evaluation
                           |
                           v
                  MLflow Registry
                           |
                           v
                    FastAPI Service
                           |
              +------------+------------+
              |                         |
              v                         v
       RUL Prediction             Fleet Prediction
              |
              v
     Conformal Prediction Intervals
              |
              v
       Streamlit Dashboard

Reference Data + Comparison Data
              |
              v
       Evidently Drift Analysis
              |
              v
        Drift Monitor Dashboard
```

## Dataset

**NASA C-MAPSS — FD001**

The project uses the FD001 subset of the Commercial Modular Aero-Propulsion System Simulation dataset.

The dataset contains simulated aircraft engine run-to-failure trajectories and sensor measurements collected over successive operational cycles.

Key characteristics:
- Multiple engine units with time-series observations.
- Operational settings and sensor measurements.
- Run-to-failure training trajectories.
- Test trajectories with corresponding RUL labels supplied separately by the dataset.

### Preprocessing pipeline

1. Load the training and test datasets.
2. Identify engine units and order observations by operational cycle.
3. Prepare sensor features for model input.
4. Apply the project's configured preprocessing and feature scaling.
5. Generate fixed-length sequences of **30 cycles**.
6. Prepare RUL targets for supervised learning.
7. Train and evaluate models using the processed sequences.

The processed sequence shapes used during development were:

| Dataset | Shape |
|---|---:|
| Training sequences | `(17731, 30, 14)` |
| Test sequences | `(100, 30, 14)` |

Each sequence contains 30 time steps and 14 sensor features.

## Models

### 1. LSTM

A Long Short-Term Memory network learns temporal dependencies in sequential engine sensor data. It serves as the primary production prediction model.

### 2. Temporal Convolutional Network (TCN)

A TCN uses temporal convolutions to learn patterns in sensor sequences. It provides an alternative deep learning architecture for comparison.

The reported TCN results were obtained from a 10-epoch training run.

### 3. XGBoost

XGBoost provides a tree-based machine learning baseline against which the sequence models can be evaluated.

## Model Evaluation

The models were evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the PHM scoring function.

| Model | RMSE | MAE | PHM Score |
|---|---:|---:|---:|
| LSTM | 12.94 | 9.63 | 256.94 |
| XGBoost | 15.02 | 11.18 | 360.76 |
| TCN | 18.92 | 14.01 | 998.23 |

Lower values are better for all three reported metrics. The LSTM achieved the best results among these recorded runs.

*Note: These are the recorded project results, not a claim that every model was trained with identical settings or compute budgets.*

## Uncertainty Estimation with Conformal Prediction

A point estimate does not communicate how uncertain a prediction may be. The project uses conformal prediction to generate an interval around the predicted RUL.

Recorded configuration and results:
- Target coverage: **90%**
- Empirical coverage: **92%**
- Conformal quantile (`q_hat`): approximately **22.27 cycles**
- Typical interval width: approximately **43.5 cycles**

The interval provides additional context around the point prediction. Coverage results apply to the evaluated data and do not guarantee the same coverage under future distribution shifts.

## MLflow: Experiment Tracking & Model Registry

MLflow is used to track model experiments and manage registered models.

Implemented capabilities include:
- Model experiment metrics and artifacts.
- Model registration.
- Model versioning.
- Production-stage model selection.
- Loading the registered LSTM model for inference.

The registered LSTM model's Production version was verified as **version 2** during development. The serving code also contains a checkpoint fallback path if a Production model is unavailable.

## FastAPI Inference Service

FastAPI exposes the prediction functionality through REST endpoints.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check service health and loaded model information |
| `POST /predict` | Predict RUL for an input sequence |
| `POST /predict/fleet` | Generate predictions for multiple engine units |
| `GET /alerts` | Retrieve fleet alerts |

Interactive API documentation is available through Swagger UI.

## Streamlit Dashboard

The dashboard provides an interface for inspecting predictions and model-monitoring information.

### Fleet Health
Displays fleet-level RUL estimates and health categories, including critical, warning, and healthy units.

### Model Performance
Compares the recorded RMSE, MAE, and PHM scores of available model architectures.

### Drift Monitor
Displays the results of reference-versus-comparison data drift analysis using Evidently.

The recorded comparison flagged **36 of 42 monitored features (85.7%)**. This result describes the particular comparison performed during development; it does not establish that live engine data is being collected.

### Retrain History
Provides an interface for viewing retraining-history information. Automatic retraining and persistent logging of every retraining run depend on the configured pipeline and should not be assumed solely from the existence of this dashboard tab.

## Drift Monitoring

The current drift demonstration compares reference training data against comparison data from the available test dataset.

```text
Reference Dataset
        |
        +---------> Evidently Comparison <---------+
                                                   |
                                      Comparison Dataset
                                                   |
                                                   v
                                          Drift Report
                                                   |
                                                   v
                                         Streamlit Display
```

This demonstrates the data-drift analysis workflow. Continuous monitoring of real engine sensors would require an additional ingestion pipeline, storage, and scheduled or streaming evaluations.

## Running the Project Locally

### Prerequisites

- Python version compatible with the project dependencies.
- Git.
- A virtual environment.
- The required project dataset and model artifacts.

### 1. Clone the repository

```bash
git clone https://github.com/saimanishreddygollapally-dev/RUL-prediction-system.git
cd RUL-prediction-system
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the repository includes `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Python path

From the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

### 5. Start the FastAPI service

Open a PowerShell terminal in the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m uvicorn pdm.serving.api:app --host 127.0.0.1 --port 8000
```

Useful URLs:

- Swagger documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

### 6. Start the Streamlit dashboard

Open a second PowerShell terminal in the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Use the local URL printed by Streamlit in the terminal, usually `http://localhost:8501`.

### 7. Run tests

Use the project's virtual environment:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pytest
```

During development, the test suite reported **23 passing tests**. Results may vary after code or dependency changes.

## Project Structure

```text
predictive-maintenance-rul-main/
├── app/
│   └── streamlit_app.py
├── src/
│   └── pdm/
│       ├── models/
│       ├── serving/
│       └── pipeline/
├── data/
│   └── processed/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

This is a simplified overview. Refer to the repository for the complete file and module structure.

## Key Engineering Highlights

- Time-series feature engineering and sequence-based RUL prediction.
- Comparison of recurrent, convolutional, and tree-based models.
- Uncertainty estimation using conformal prediction.
- Versioned model management through MLflow.
- REST-based inference and fleet-level predictions.
- Data-drift analysis and interactive monitoring.
- Automated testing and modular Python components.

## Limitations & Future Improvements

- Integrate a live sensor simulator or an actual sensor-data source.
- Persist retraining records with timestamps, metrics, model versions, and outcomes.
- Implement a validated retraining trigger and candidate-model evaluation workflow.
- Promote a new model only after it meets defined validation criteria.
- Add authentication and access control for deployed services.
- Monitor model performance and prediction quality over time.
- Deploy the API and dashboard in a production environment.

## Disclaimer

This project is an educational predictive-maintenance prototype built using simulated engine data. Its predictions are not certified for real aircraft maintenance or operational safety decisions.

## Author

**Sai Manish Reddy Gollapally**

GitHub: [saimanishreddygollapally-dev](https://github.com/saimanishreddygollapally-dev)

Repository: [RUL-prediction-system](https://github.com/saimanishreddygollapally-dev/RUL-prediction-system)
