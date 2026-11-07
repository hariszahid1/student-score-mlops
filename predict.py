# predict.py - predict score for a new student (no training)
import pandas as pd
import joblib

model = joblib.load("model.pkl")
new_student = pd.DataFrame({"study_hours": [6], "attendance_pct": [90]})
score = model.predict(new_student)[0]
print(f"6 study hours, 90% attendance -> predicted score: {score:.1f}")
