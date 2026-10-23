
import mlflow
from mlflow.tracking import MlflowClient
from pdm.config import PROJECT_ROOT

mlflow.set_tracking_uri(
    "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
)
client = MlflowClient()

for experiment in client.search_experiments():
    print(f"\nExperiment: {experiment.name} (ID={experiment.experiment_id})")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=20,
        order_by=["attributes.start_time DESC"],
    )

    for run in runs:
        print(
            f"  Run={run.info.run_id[:8]}, "
            f"status={run.info.status}, "
            f"metrics={run.data.metrics}"
        )
