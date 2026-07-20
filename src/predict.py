import numpy as np
import requests
import pandas as pd


def predict():

    # Importing the dataset + preprocessing
    df = rp.conso_preprocess(Path("../data/conso/real_time_conso/"))
    df = df[df["Heures"].apply(lambda x: x.minute in {00, 30})]
    df = df.reset_index(drop=True)
    df.loc[len(df)] = None
    t = df["Heures"].iloc[-2]  
    new_time = (datetime.combine(datetime.today(), t) + timedelta(minutes=30)).time()

    # Adding lagged consumtion features (lag-1, lag-2, lag-48...)
    df = fe.lagged_consumption(df)
    df.loc[len(df)-1, "Heures"] = new_time
    
    
    return None