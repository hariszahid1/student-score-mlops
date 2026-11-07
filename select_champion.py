# select_champion.py - lowest MAE becomes MLflow alias "champion"
import joblib
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

versions = client.search_model_versions("name='student-score-model'")
if not versions:
    raise SystemExit("No versions found. Run train.py first.")

best = None
best_mae = float("inf")
for v in versions:
    run = client.get_run(v.run_id)
    mae = run.data.metrics.get("mae")
    model_type = run.data.params.get("model_type", "?")
    print(f"  v{v.version}: model_type={model_type}  mae={mae}")
    if mae is not None and mae < best_mae:
        best_mae = mae
        best = v

if best is None:
    raise SystemExit("No MAE metrics found.")

client.set_registered_model_alias("student-score-model", "champion", best.version)
print(f"Champion set to v{best.version} (MAE={best_mae:.4f})")

model = mlflow.sklearn.load_model("models:/student-score-model@champion")
joblib.dump(model, "champion.pkl")
print("Champion model saved to champion.pkl")
