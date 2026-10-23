
import mlflow
from mlflow.tracking import MlflowClient
from pdm.config import PROJECT_ROOT

db_path = PROJECT_ROOT / "mlflow.db"
tracking_uri = "sqlite:///" + db_path.as_posix()

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

print("Tracking URI:", tracking_uri)
print("\nRegistered models:")

models = client.search_registered_models()

if not models:
    print("No registered models found.")

for model in models:
    print(f"\nModel: {model.name}")
    versions = client.search_model_versions(f"name='{model.name}'")

    for version in versions:
        print(
            f"Version: {version.version}, "
            f"Stage: {version.current_stage}, "
            f"Run ID: {version.run_id}"
        )
