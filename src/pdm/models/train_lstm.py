import mlflow
import mlflow.pytorch
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from pdm.config import FEATURE_SENSORS, PROC, PROJECT_ROOT, RANDOM_STATE
from pdm.evaluate.metrics import mae, phm_score, rmse
from pdm.models.lstm import RULLSTM


def main(subset="FD001", epochs=40, bs=256, lr=1e-3):
    torch.manual_seed(RANDOM_STATE)
    d = np.load(PROC / f"{subset}.npz")
    keys = ("Xtr", "ytr", "Xte", "yte")
    Xtr, ytr, Xte, yte = (torch.tensor(d[k], dtype=torch.float32) for k in keys)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    model = RULLSTM(len(FEATURE_SENSORS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Set MLflow tracking URI to use project-relative SQLite database
    mlflow_db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

    mlflow.set_experiment("rul-cmapss")

    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params(dict(model="LSTM", epochs=epochs, bs=bs, lr=lr, seq_len=Xtr.shape[1]))
        for ep in range(epochs):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte.to(dev)).cpu().numpy()
        mlflow.log_metrics({"rmse": rmse(yte.numpy(), pred), "mae": mae(yte.numpy(), pred),
                            "phm_score": phm_score(yte.numpy(), pred)})
        torch.save(model.state_dict(), PROC / "lstm.pt")
        mlflow.log_artifact(str(PROC / "lstm.pt"))

        # Log the trained PyTorch model to MLflow Registry
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="rul-lstm",
        )

        print("LSTM RMSE", rmse(yte.numpy(), pred))


if __name__ == "__main__":
    main()
