from fastapi import FastAPI
from pathlib import Path
from joblib import load
import numpy as np
import pandas as pd
import psycopg
import pytz
from datetime import datetime
from sklearn.linear_model import LinearRegression

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
    current_date = datetime.now(france_tz).date()
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT timestamp, consumption_mw
            FROM forecasts
            WHERE timestamp::date = (%s)
            ORDER BY timestamp;

        """, (current_date,))
    
        predictions = cursor.fetchall()
    
        cursor.execute("""
            SELECT timestamp, hist_consumption_mw
            FROM historical
            WHERE timestamp::date = (%s)
            ORDER BY timestamp;
        """, (current_date,))

        historical = cursor.fetchall()
        
    conn.close()

    return {
        "historical": historical,
        "predictions": predictions
    }




@app.get("/demand")
def get_peak():

    france_tz = pytz.timezone("Europe/Paris")
    current_date = datetime.now(france_tz).date()
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
            WHERE timestamp::date = (%s)
            ORDER BY timestamp;

        """, (current_date,))
        predictions = cursor.fetchall()

        cursor.execute("""
            SELECT timestamp, hist_consumption_mw
            FROM historical
            WHERE timestamp::date = (%s)
            ORDER BY timestamp;
        """, (current_date,))
        historical = cursor.fetchall()


        # The dict of all the infos
        all_infos = dict()
        x = np.arange(len(predictions)).reshape(-1, 1)

        # If there are no forecast data in the db, we indicate it in the dict all_infos
        # Otherwise we fill the dictionnary as usuall
        if len(x) == 0:
            all_infos["ready"] = False
        else :
        # Else, we start the analysis of the preds


            
            y = np.array(pd.DataFrame(predictions).iloc[:, 1])
            # The highest value predicted + precentage increase
            increase_infos = tuple()
            percentage_increase = 0
            
            df_pred = pd.DataFrame(np.array(predictions), columns=["time", "pred"])
            df_pred = df_pred.sort_values(by="time", ascending=False)

            # If the minute number is between [10, 40[ we select the predictions older than 'current_hour:30'
            # Otherwise, if it's  between [0, 9] or [40, 59] we select the predictons older than 'current_hour:00'
            current_time = datetime.now(france_tz).time()
            if (10 <= current_time.minute) and (current_time.minute < 40):
                current_time = current_time.replace(minute=30, second=0, microsecond=0)
            else:
                current_time = current_time.replace(hour= current_time.hour + 1, minute=00, second=0, microsecond=0)
            df_pred = df_pred[df_pred["time"].dt.time >= current_time]


            
            # Sometimes in dev df_pred can be empty because we have predictions older than the current hour
            if len(df_pred) == 0:
                all_infos["ready"] = False
                
            else : 
            # Else, we continue the analysis of the preds
                
                
                all_infos["ready"] = True

                #df_pred = df_pred.iloc[:10]
                df_pred = df_pred.sort_values(by="pred")
                df_pred = df_pred.reset_index(drop=True)
        
                y_max = df_pred["pred"].iloc[-1]
                time_y_max = df_pred["time"].iloc[-1]
                time_y_max = time_y_max.strftime('%H:%M')
        
                y_0 = None
                time_y_0 = None
                # IF THE HISTORICAL DB IS EMPTY DF_HIST WILL RAISE AN EXCEPTION !
                if len(historical) == 0:
                    increase_infos = (False, 0, 0, 0)
                    
                else :
                    # We manage the case when there are no historical data saved in the db
                    df_hist = pd.DataFrame(np.array(historical), columns=["time", "hist"])
                    # if the historical db is empty for some reason, it would be impossible to get the last known historical point
                    # So we replace that point by the nearest prediction given by the model
                    if len(df_hist) == 0:
                        df_pred_temp = df_pred.sort_values(by="time")
                        y_0 = df_pred_temp["pred"].iloc[0]
                        time_y_0 = df_pred_temp["time"].iloc[0]
                    else:
                        df_hist = df_hist.sort_values(by="time")
                        y_0 = df_hist["hist"].iloc[-1]
                        time_y_0 = df_hist["time"].iloc[-1]
        
                    if y_max > y_0:
                        percentage_increase = float((((y_max - y_0) / y_0)*100))
                        increase_infos = (True, y_max, time_y_max, percentage_increase)
            
                    else : 
                        increase_infos = (False, y_0, time_y_0, percentage_increase)
        
                # (Boolean, y_max, time, percentage_increase)
                all_infos["highest_value"] = increase_infos
        
        
                # Trend informations
                # 3 different levels of electricity demand : Low, Medium, Hard
                # We fit a function on our predictions to get the trend of the electricity consumption
                model = LinearRegression()
                model.fit(x, y)
        
        
                trend = model.predict(x)
                slope = model.coef_[0]
        
                elec_demand_infos = tuple()
                if slope > 4000:
                    elec_demand_infos = (slope, "High")
                if slope > 1500:
                    elec_demand_infos = (slope, "Medium")
                else:
                    elec_demand_infos = (slope, "Low")
                # (Slope rate, elec demand level)
                all_infos["slope"] = elec_demand_infos

                
        
                # Spike Detection

                y = list(df_pred["pred"])  # We add to y (the list of the next predictions) the last known historical value
                y = np.insert(y, 0, y_0)
                diffs = np.diff(y)
                idx_max_diff = np.argmax(diffs)
                #print(idx_max_diff)
                df_pred = df_pred.sort_values(by="time").reset_index(drop=True)
                #print(df_pred)
                #print(df_pred["pred"].iloc[idx_max_diff])
                y_spike = df_pred["pred"].iloc[idx_max_diff]
                y_spike_time = df_pred["time"].iloc[idx_max_diff]
        
                spike_infos = tuple()
                if diffs.max() > 4500:
                    spike_infos = (True, y_spike, y_spike_time)
                else : 
                    spike_infos = (False, 0, 0)
        
                all_infos["spike"] = spike_infos
        
    
        return all_infos
    
    
    
            
        

        


        
        

    
        
    
        
