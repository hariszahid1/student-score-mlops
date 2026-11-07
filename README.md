# Student Score MLOps

Predict exam **score** from `study_hours` + `attendance_pct`.

Same pipeline as the house-price lab: DVC → S3, scikit-learn + MLflow (MAE), FastAPI, pytest, GitHub Actions → Docker Hub.

| Guide (houses) | This project |
|----------------|--------------|
| `houses.csv` | `students.csv` |
| `price_lakh` | `score` |
| `house-price-project` | `student-score-project` |
| `house-api` | `student-score-api` |

**S3:** `s3://aqsa-mlops-dvc-2026/dvc`  
**Docker Hub:** `harris944/student-score-api`  
**GitHub:** `hariszahid1/student-score-mlops`
