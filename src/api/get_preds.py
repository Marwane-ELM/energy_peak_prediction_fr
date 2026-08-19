from fastapi import FastAPI
from pathlib import Path
from joblib import load
import psycopg

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
    
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT timestamp, consumption_mw
            FROM forecasts
            WHERE timestamp::date = CURRENT_DATE
            ORDER BY timestamp;

        """)
    
        forecast = cursor.fetchall()

        cursor.execute("""
            SELECT timestamp, hist_consumption_mw
            FROM historical
            WHERE timestamp::date = CURRENT_DATE
            ORDER BY timestamp;
        """)

        historical = cursor.fetchall()
        
    conn.close()

    return {"forecast" : forecast, "historical" : historical}
