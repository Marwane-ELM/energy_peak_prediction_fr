from pathlib import Path
import numpy as np
import pandas as pd
import os.path
from datetime import datetime


def split_hour_date(df):
    """
    Input : the best dataset among the 3 created versions 
    """
    df = df.copy()
    df["Date"]   = pd.to_datetime(df["Date"])
    df["Heures"] = pd.to_timedelta(df["Heures"].astype(str))
    df["full_date"] = df["Date"] + df["Heures"]
    # We extract the data about the time (date adnd hour) by creating new columns. 
    #(Some models will not be able to be trained on a dataset having columns containing "datetime" format)
    df["year"] = df["full_date"].dt.year
    df["month"] = df["full_date"].dt.month
    df["day"] = df["full_date"].dt.day
    df["hour"] = df["full_date"].dt.hour
    df["minute"] = df["full_date"].dt.minute
    
    df = df.drop(["Date", "Heures", "full_date"], axis=1)
    return df
    
    

def days(df):
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday,..., 6=Sunday
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    return None
