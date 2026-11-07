# Student Score MLOps

Predict exam **score** from `study_hours` + `attendance_pct`.

Same pipeline as the house-price lab: DVC → S3, scikit-learn + MLflow (MAE), FastAPI, pytest, GitHub Actions → Docker Hub.

| Guide (houses) | This project |
|----------------|--------------|
| `houses.csv` | `students.csv` |
| `price_lakh` | `score` |
| `house-price-project` | `student-score-project` |
| `house-api` | `student-score-api` |

**Champion:** linear model (MAE ≈ 0.79)

**S3 (DVC):** `s3://haris-mlops-dvc-2026/aqsa-mlops-dvc-2026/dvc`  
*(Target bucket name `aqsa-mlops-dvc-2026` — IAM currently allows writes under `haris-mlops-dvc-2026`; remote `aqsa-target` points at the dedicated bucket once policy includes it.)*

**Docker Hub:** https://hub.docker.com/r/harris944/student-score-api  
**GitHub:** https://github.com/hariszahid1/student-score-mlops
