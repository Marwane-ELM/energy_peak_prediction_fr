import numpy as np
import requests
import pandas as pd


def predict():

    # Importing the dataset + preprocessing
    df = rp.conso_preprocess(Path("../data/conso/real_time_conso/"))
    # We only keep the rows with minutes = 00 or 30
    df = df[df["Heures"].apply(lambda x: x.minute in {00, 30})]
    df = df.reset_index(drop=True)
    df = df.drop(["Consommation", "hour", "month", "day_of_week"], axis=1)

    # We add the date and hour
    df.loc[len(df)] = None
    t = df["Heures"].iloc[-2]  
    new_time = (datetime.combine(datetime.today(), t) + timedelta(minutes=30)).time()
    df.loc[len(df)-1, "Heures"] = new_time
    df.loc[len(df)-1, "Date"] = datetime.today().strftime('%Y-%m-%d')
    df = fe.date_and_hour(df)

    # Adding lagged consumtion features (lag-1, lag-2, lag-48...)
    df = fe.lagged_consumption(df)

    #Adding the infos about the holidays
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

    # Adding other columns
    df = fe.date_and_hour(df)
    df = fe.cyclical_encoding(df)
    df = fe.rolling_window(df)
    df = fe.lagged_trend(df)
    df = df.drop(["Consommation", "hour", "month", "day_of_week"], axis=1)
    
    
    return None