
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from pdm.serving.registry import (
    PROJECT_ROOT,
    PROC,
    DEVICE,
    FEATURE_SENSORS,
    RULLSTM,
)

# Use the same MLflow database as the project.
mlflow_db_path = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(
    "sqlite:///" + mlflow_db_path.as_posix()
)
mlflow.set_experiment("rul-cmapss")

checkpoint_path = PROC / "lstm.pt"

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Existing LSTM checkpoint not found: {checkpoint_path}"
    )

# Load the already-trained model. No training occurs.
model = RULLSTM(n_features=len(FEATURE_SENSORS))
state_dict = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# Verify the loaded model produces a valid prediction.
example_input = torch.zeros(
    1, 30, len(FEATURE_SENSORS),
    dtype=torch.float32,
    device=DEVICE,
)

with torch.no_grad():
    prediction = model(example_input)

print("Checkpoint loaded successfully.")
print("Test output shape:", tuple(prediction.shape))

# Register the existing model in MLflow.
with mlflow.start_run(run_name="register-existing-lstm") as run:
    mlflow.log_param("model", "LSTM")
    mlflow.log_param("checkpoint_reused", True)
    mlflow.log_param("sequence_length", 30)
    mlflow.log_param("n_features", len(FEATURE_SENSORS))

    mlflow.pytorch.log_model(
    pytorch_model=model,
    name="model",
    registered_model_name="rul-lstm",
    input_example=example_input.detach().cpu().numpy(),
    serialization_format="pickle",
    export_model=False,
	)

    print("MLflow run ID:", run.info.run_id)

client = mlflow.MlflowClient()

versions = client.search_model_versions(
    "name = 'rul-lstm'"
)

if not versions:
    raise RuntimeError(
        "Registration did not create a model version."
    )

latest = max(versions, key=lambda v: int(v.version))

print("\nRegistration successful.")
print("Registered model: rul-lstm")
print("Version:", latest.version)
print("Run ID:", latest.run_id)
print("Stage:", latest.current_stage)
print("\nThe model has NOT been promoted to Production yet.")
