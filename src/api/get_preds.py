from fastapi import FastAPI
from pathlib import Path
from joblib import load
import psycopg
import pytz
from datetime import datetime

app = FastAPI()

@app.get("/predict")
def get_preds():
    conn = psycopg.connect(
        host="localhost",
        port=5432,
        dbname="energy_db",
        user="postgres",
        password="postmdp"
    )

    france_tz = pytz.timezone("Europe/Paris")
    current_time = datetime.now(france_tz).date()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT timestamp, consumption_mw
            FROM forecasts
            WHERE timestamp::date = (%s)
            ORDER BY timestamp;

        """, (current_time,))
    
        predictions = cursor.fetchall()

        cursor.execute("""
            SELECT timestamp, hist_consumption_mw
            FROM historical
            WHERE timestamp::date = (%s)
            ORDER BY timestamp;
        """, (current_time,))

        historical = cursor.fetchall()
        
    conn.close()

    return {
        "historical": historical,
        "predictions": predictions
    }