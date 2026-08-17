from fastapi import FastAPI
from pathlib import Path
from joblib import load
import psycopg

app = FastAPI()

path_preds = Path("../../artifacts/model_artifacts/predictions/preds.joblib")

@app.get("/predict")
def get_preds():
    conn = psycopg.connect(
        host="localhost",
        port=5432,
        dbname="energy_db",
        user="postgres",
        password="postmdp"
    )
    
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT timestamp, consumption_mw
            FROM forecasts
            ORDER BY timestamp
        """)
    
        data = cursor.fetchall()

    conn.close()

    
    return data
    