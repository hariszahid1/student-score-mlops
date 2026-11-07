# export_model.py - load champion from MLflow into champion.pkl
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/student-score-model@champion")
joblib.dump(model, "champion.pkl")
print("Champion model saved to champion.pkl")
