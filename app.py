# app.py - FastAPI for student exam score prediction
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Student Score API")
model = joblib.load("champion.pkl")


class Student(BaseModel):
    study_hours: float = Field(gt=0, description="Hours studied")
    attendance_pct: float = Field(gt=0, le=100, description="Attendance percent")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(student: Student):
    data = pd.DataFrame(
        [{"study_hours": student.study_hours, "attendance_pct": student.attendance_pct}]
    )
    score = model.predict(data)[0]
    return {"score": round(float(score), 1)}
