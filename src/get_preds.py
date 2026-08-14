from fastapi import FastAPI
from pathlib import Path
from joblib import load

app = FastAPI()

path_preds = Path("../artifacts/model_artifacts/predictions/preds.joblib")

@app.get("/predict")
def get_preds():
    dates, preds = load(path_preds)
    return dates, preds
    