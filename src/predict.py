import numpy as np
import requests
import pandas as pd

import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry

import os
import sys
sys.path.append(os.path.abspath(".."))

from datetime import date, datetime, timedelta
from vacances_scolaires_france import SchoolHolidayDates

from pathlib import Path
from joblib import load, dump
import src.raw_preprocessing as rp
import src.feature_engineering as fe

def predict():

#----- Importing the dataset + preprocessing ----
    df = rp.conso_preprocess(Path("../data/conso/real_time_conso/"))
    # We only keep the rows with minutes = 00 or 30
    df = df[df["Heures"].apply(lambda x: x.minute in {00, 30})]
    df = df.reset_index(drop=True)

#----- We add the date and hour -----
    df.loc[len(df)] = None
#----- Adding lagged consumtion features (lag-1, lag-2, lag-48...)-----
    df = fe.lagged_consumption(df)
    
    t = df["Heures"].iloc[-2]  
    new_time = (datetime.combine(datetime.today(), t) + timedelta(minutes=30)).time()
    
    df.loc[len(df)-1, "Heures"] = new_time
    df.loc[len(df)-1, "Date"] = datetime.today().strftime('%Y-%m-%d')


#----- Adding the infos about the holidays -----
    today = date.today().isoformat()
    url = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"
    
    zones = ['A', 'B', 'C']
    vacances = ['vacances de la toussaint', 'vacances de noël', "vacances d'hiver",'vacances de printemps', "vacances d'été"]
    feries = [
        "jour de l'an",
        "lundi de pâques",
        "fête du travail",
        "victoire 1945",
        "ascension",
        "lundi de pentecôte",
        "fête nationale",
        "assomption",
        "toussaint",
        "armistice",
        "noël",
        "pont de l'ascension",
    ]
    
    #finished_with_holidays = False
    df.loc[:, "Zone_A"] = 0
    df.loc[:, "Zone_B"] = 0
    df.loc[:, "Zone_C"] = 0
    
    df.loc[:, "public_holidays"] = 0
    df.loc[:, "Vacances de la Toussaint"] = 0
    df.loc[:, "Vacances de Noël"] = 0
    df.loc[:, "Vacances d'Hiver"] = 0
    df.loc[:, "Vacances de Printemps"] = 0
    df.loc[:, "Vacances d'Été"] = 0
    
    for z in zones:
        params = {
            "where": f"start_date <= date'{today}' AND end_date >= date'{today}' AND zones = 'Zone {z}'"
        }
        response = requests.get(url, params=params).json()
        # If no holidays we set the columns with the value 0
        if response["total_count"] == 0:
            continue
    
        else : 
            
            for event in response["results"]:
                # We check if it's school holidays
                if event["description"].lower() in vacances:
                    df.loc[:, f"Zone_{z}"] = 1
                    if event["description"].lower() == "vacances de la toussaint":
                        df.loc[:, "Vacances de la Toussaint"] = 1
                        
                    elif event["description"].lower() == "vacances de noël":
                        df.loc[:, "Vacances de Noël"] = 1
                        
                    elif event["description"].lower() == "vacances d'hiver":
                        df.loc[:, "Vacances d'Hiver"] = 1
                        
                    elif event["description"].lower() == "vacances de printemps":
                        df.loc[:, "Vacances de Printemps"] = 1
    
                    elif event["description"].lower() == "vacances d'été":
                        df.loc[:, "Vacances d'Été"] = 1
                # Or if it's public holidays (jours fériés)
                elif event in feries:
                    df.loc[:, "public_holidays"] = 1

    
    #-----  Adding other columns -----
    df = fe.date_and_hour_pred(df)
    df = fe.cyclical_encoding(df)
    df = fe.rolling_window(df)
    df = fe.lagged_trend(df)
    df = fe.seasons_linear(df)
    df = df.drop(["Consommation"], axis=1)

    # --- Input dataframe ---
    row = df.iloc[-1]
    pred = pd.DataFrame([row] * 10)
    pred["full_date"] = row["full_date"] + pd.to_timedelta(range(10), unit="m") * 30
    pred = pred.reset_index(drop=True)


# ---- Open-Meteo -----
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    lat_long = {
        '13': {'ville': 'Marseille', 'latitude': 43.2965, 'longitude': 5.3698},     # Bouches-du-Rhône
        '33': {'ville': 'Bordeaux',  'latitude': 44.8378, 'longitude': -0.5792},    # Gironde
        '44': {'ville': 'Nantes',    'latitude': 47.2184, 'longitude': -1.5536},    # Loire-Atlantique
        '59': {'ville': 'Lille',     'latitude': 50.6292, 'longitude': 3.0573},     # Nord
        '69': {'ville': 'Lyon',      'latitude': 45.7640, 'longitude': 4.8357},     # Rhône
        '75': {'ville': 'Paris',     'latitude': 48.8566, 'longitude': 2.3522},     # Paris
    }
    
    station_population = {
        '13': 2087658,   # Bouches-du-Rhône 
        '33': 1690493,   # Gironde           
        '44': 1487570,   # Loire-Atlantique   
        '59': 2615635,   # Nord              
        '69': 1914667,   # Rhône             
        '75': 2103778,   # Paris
    }
    
    total_pop = sum(station_population.values())
    weights = {city: pop / total_pop for city, pop in station_population.items()}
    
    
    cols = ['T', 'U', 'FF', 'PMER', 'RR1']
    df_temp = pd.DataFrame(
        np.zeros(shape=(pred.shape[0], len(cols))),
        columns=cols
    )
    
    for k, v in lat_long.items():
    
        # Calling the API for our department
        params = {
        	"latitude": v["latitude"],
        	"longitude": v["longitude"],
        	"hourly": ["temperature_2m", "relative_humidity_2m", "rain", "surface_pressure", "wind_speed_10m"],
        	"timezone": "Europe/London",
        	"past_days": 7,
        	"forecast_days": 1,
        }
        responses = openmeteo.weather_api(url, params = params)
        
        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
    
        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_rain = hourly.Variables(2).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(3).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
        
        hourly_data = {
        	"date": pd.date_range(
        		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        		freq = pd.Timedelta(seconds = hourly.Interval()),
        		inclusive = "left"
        	).tz_convert(response.Timezone().decode())
        }
    
        # Preprocessing the weather date
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
        hourly_data["rain"] = hourly_rain
        hourly_data["surface_pressure"] = hourly_surface_pressure
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        
        hourly_dataframe = pd.DataFrame(data = hourly_data)
    
        hourly_dataframe = hourly_dataframe.rename(columns={
            "temperature_2m" : "T",
            "relative_humidity_2m": "U",
            "rain" : "RR1",
            "surface_pressure" : "PMER",
            "wind_speed_10m" : "FF"
        })
    
        # We add the 30min weather date 
        hourly_dataframe["date"] = pd.to_datetime(hourly_dataframe["date"])
        hourly_dataframe = hourly_dataframe.set_index("date")
        hourly_dataframe = hourly_dataframe.asfreq("30min")
    
        # We interpolate the missing values
        hourly_dataframe["RR1"] = hourly_dataframe["RR1"].fillna(0)
        hourly_dataframe = hourly_dataframe.interpolate(method="time")
        hourly_dataframe = hourly_dataframe.reset_index()
        hourly_dataframe["date"] = hourly_dataframe["date"].dt.tz_localize(None)
        hourly_dataframe = hourly_dataframe[hourly_dataframe["date"].isin(pred["full_date"])]
        hourly_dataframe = hourly_dataframe.reset_index()
        
        pred[f"{k}T"] = hourly_dataframe[['T']]
    
    
        population = weights[k]
        df_temp += hourly_dataframe[cols].values * population
    
    
    pred = pd.concat([pred, df_temp], axis=1)
    pred = pred.drop("full_date", axis=1)

    
# -- Interactions features tailored for linear models ---
    pred = fe.interactions_linear(pred)
        
    
    # Making predictions by loading the models
    models = load("../artifacts/model_artifacts/Ridge_2026-07-30_23-24-53/Ridge_models.joblib")
    
    pred2 = pred.copy()
    pred2 = fe.drop_useless(pred2)
    
    predictions = {}
    for i in range(0, 10):
        if i == 0:
            p = models[f"Ridge_{i}"].predict(pred)
            predictions[f"horizon_{i}"] = p[0]
        else:
            p = models[f"Ridge_{i}"].predict(pred2.iloc[i-1:i])
            predictions[f"horizon_{i}"] = p[0]
    
    
    print(predictions)
    path_preds = Path("../artifacts/model_artifacts/predictions/preds.joblib")
    if path_preds.exists():
        os.remove(path_preds)
        
    dump(predictions, "../artifacts/model_artifacts/predictions/preds.joblib",)


if __name__ == "__main__":
    predict()