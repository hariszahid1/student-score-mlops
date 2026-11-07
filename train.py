# train.py - predict student exam score from study hours + attendance
#
# Usage:  python train.py          -> LinearRegression
#         python train.py tree     -> DecisionTree
#         python train.py forest   -> RandomForest
import sys
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("student-score-project")

model_type = sys.argv[1] if len(sys.argv) > 1 else "linear"
test_size = 0.2

data = pd.read_csv("students.csv")
print("Loaded", len(data), "students")

X = data[["study_hours", "attendance_pct"]]
y = data["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
print("Training on", len(X_train), "students, testing on", len(X_test), "students")

with mlflow.start_run():
    if model_type == "tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == "forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    guesses = model.predict(X_test)
    mae = mean_absolute_error(y_test, guesses)
    print(f"Model: {model_type}  |  MAE (average mistake): {mae:.2f} points")

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("rows_in_dataset", len(data))
    mlflow.log_metric("mae", mae)

    with open("metrics.json", "w") as f:
        json.dump({"mae": mae}, f)

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="student-score-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl and recorded in MLflow")
