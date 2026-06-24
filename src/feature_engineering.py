from pathlib import Path
import numpy as np
import pandas as pd
import os.path
from datetime import datetime
import src.raw_preprocessing as rp


def date_and_hour(df):
    """
    Input : the best dataset among the 3 created versions 
    """
    df["Date"]   = pd.to_datetime(df["Date"])
    df["Heures"] = pd.to_timedelta(df["Heures"].astype(str))
    df["full_date"] = df["Date"] + df["Heures"]
    # We extract the data about the time (date adnd hour) by creating new columns. 
    #(Some models will not be able to be trained on a dataset having columns containing "datetime" format)
    df["year"] = df["full_date"].dt.year
    df["month"] = df["full_date"].dt.month

    # We only keep the decimal value of the time (we don"t split the hour and minute cuz it"s useless)
    df["hour"] = df["full_date"].dt.hour + (df["full_date"].dt.minute / 60)

    # Here we add extra columns that"ll give us more precise info about the current day
    df["day_of_week"] = df["full_date"].dt.dayofweek   # 0=Monday,..., 6=Sunday
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    
    df = df.drop(["Date", "Heures", "full_date"], axis=1)
    return df
    

def cyclical_encoding(df):
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    
    df["day_of_week_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df

def lagged_consumption(df, horizon_shift=0):
    
    """We add the horizon_shift to the lagged values in order to do multi-steps forecasting.
    Indeed, if we want to predict the energy consumption in 5 hours (10 rows of 30 min timestamp), we will not be
    able to get the same lag data as the next 30 min timestamp. 
    For our example we should add (5 * 2 = 10) to the lagged values. 
    """
    lagged_values = [1, 2, 48, 336]  # last 30 min, hour, day (at same hour) and last week
    for l in lagged_values:
        df.loc[:, f"lagged_{l}"] = df["Consommation"].shift(l + horizon_shift)

    return df



def rolling_window(df, horizon_shift=0):
    shift = 1 + horizon_shift  # shift(1) for one-step and shift(1+h) for multi-step
    conso = df["Consommation"].shift(shift)
    df.loc[:, "rolling_mean_24h"] = conso.rolling(48).mean()
    df.loc[:, "rolling_std_24h"] = conso.rolling(48).std()
    df.loc[:, "rolling_mean_7d"] = conso.rolling(336).mean()
    df.loc[:, "rolling_std_7d"] = conso.rolling(336).std()
    df.loc[:, "rolling_max_24h"] = conso.rolling(48).max()
    df.loc[:, "rolling_min_24h"] = conso.rolling(48).min()
    return df



def lagged_trend(df, horizon_shift=0):
    conso_shifted = df["Consommation"].shift(1 + horizon_shift)
    
    df.loc[:, "consumption_diff_1"] = conso_shifted.diff(1)
    df.loc[:, "consumption_diff_48"] = conso_shifted.diff(48)
    df.loc[:, "consumption_pct_change_1"] = conso_shifted.pct_change(1)
    df.loc[:, "consumption_pct_change_48"] = conso_shifted.pct_change(48)
    return df

#def shift_target(target, horizon_shift=0):
    #target = target.shift(-)


def get_season(month):
    if month in [12, 1, 2]:  
        return "Winter"
    elif month in [3, 4, 5]: 
        return "Spring"
    elif month in [6, 7, 8]: 
        return "Summer"
    else:                     
        return "Autumn"

def seasons_tree(df):
    df["season"] = df["month"].apply(get_season)

    # We One-Hot encode the season column, it creates new columns for each season
    season_dummies = pd.get_dummies(df["season"], prefix="season", dtype = int)
    
    df = pd.concat([df, season_dummies], axis=1)
    df = df.drop(columns=["season"])
    
    return df

def seasons_linear(df):
    df["season"] = df["month"].apply(get_season)

    # We One-Hot encode the season column, it creates new columns for each season
    # We drop the first column to avoid multi collinearity ofr linear models
    season_dummies = pd.get_dummies(df["season"], prefix="season", dtype = int, drop_first=True)
    
    df = pd.concat([df, season_dummies], axis=1)
    df = df.drop(columns=["season"])
    
    return df


def interactions_linear(df):
    """
    This functions creates all the interactions for linear models.
    """
    
    t = df['T']  # Weighted temperature column
    u = df['U'] # Humidity column
    ff = df['FF']  # Wind speed column

    # Polynomial
    df['temp_sq'] = t ** 2
    df['humidity_sq'] = u ** 2

    # Interaction Temporal × Temporal
    df['hour_x_is_weekend'] = df['hour'] * df['is_weekend']
    df['hour_x_is_holiday'] = df['hour'] * df['public_holidays']
    df['hour_x_dow'] = df['hour'] * df['day_of_week']
    df['hour_x_month'] = df['hour'] * df['month']
    df['is_weekend_x_month'] = df['is_weekend'] * df['month']
    df['is_holiday_x_month'] = df['public_holidays'] * df['month']

    # Interaction Temporal × Weather
    df['hour_x_temp'] = df['hour'] * t
    df['hour_x_humidity'] = df['hour'] * u
    df['hour_x_wind'] = df['hour'] * ff
    df['is_weekend_x_temp'] = df['is_weekend'] * t
    df['is_holiday_x_temp'] = df['public_holidays'] * t
    df['month_x_temp'] = df['month'] * t
    df['dow_x_temp'] = df['day_of_week'] * t
    df['hour_x_temp_sq'] = df['hour'] * df['temp_sq']
    df['is_weekend_x_temp_sq'] = df['is_weekend'] * df['temp_sq']

    # Interaction Season × Weather 
    season_cols = [col for col in df.columns if "season" in col]
    for season in season_cols:
        df[f'{season}_x_temp'] = df[season] * t

    # Interaction Temporal × Season (one column per season)
    for season in season_cols:
        df[f'hour_x_{season}'] = df['hour'] * df[season]
        df[f'is_weekend_x_{season}'] = df['is_weekend'] * df[season]
        df[f'is_holiday_x_{season}'] = df['public_holidays'] * df[season]
        
    # Interaction Weather × Weather
    df['temp_x_humidity'] = t * u
    df['temp_x_wind'] = t * ff
    df['humidity_x_wind'] = u * ff


    # We add the Heating Degree Days (HDD) and Cooling Degree Days (CDD)
    base_temp = 18  # standard for France
    df["HDD"] = (base_temp - df["T"]).clip(lower=0)  # heating need
    df["CDD"] = (df["T"] - base_temp).clip(lower=0)  # cooling need
        
    df = drop_useless(df)
    return df


def interactions_tree(df):
    """This functions creates only the interactions that trees can't learn correctly"""
    
    t = df['T']
    u = df['U']
    ff = df['FF']
    # Polynomial (trees approximate this poorly)
    df['temp_sq'] = t ** 2

    # Physical combinations (continuous × continuous)
    df['temp_x_humidity'] = t * u
    df['temp_x_wind'] = t * ff

    # Complex temporal interactions (helps convergence speed)
    df['hour_x_is_weekend'] = df['hour'] * df['is_weekend']
    df['hour_x_is_holiday'] = df['hour'] * df['public_holidays']

    df = df.dropna()

    return df

def drop_useless(df):
    df = df.drop(['hour', 'month', 'day_of_week'], axis=1)
    df = df.dropna()
    return df



def feature_engineering(df, num_version, PATH_FILES, PATH_TO_SAVE, PATH_SAVE_LINEAR, PATH_SAVE_TREE):
    """
    The path should also include the name of the file (dir1/dir2/conso_v1_linear.parquet)
    """

    if num_version == 1:
        df = rp.dataset_v1(df, PATH_FILES, PATH_TO_SAVE)
    elif num_version == 2:
        df = rp.dataset_v2(df, PATH_FILES, PATH_TO_SAVE)
    elif num_version == 3:
        df = rp.dataset_v3(df, PATH_FILES, PATH_TO_SAVE)
    else : 
        print("Enter a valid number among 1, 2 and 3")
        return None


    datasets = [None, None]  # (df_linear, df_tree)
    output_path_linear = Path(PATH_SAVE_LINEAR)
    output_path_tree = Path(PATH_SAVE_TREE)

    if output_path_linear.exists() and output_path_tree.exists():
        print(f"{output_path_tree} and {output_path_linear} already exist")
        df_linear = pd.read_parquet(output_path_linear)
        df_tree = pd.read_parquet(output_path_tree)
        datasets[0], datasets[1] = df_linear, df_tree

    else :
        df = date_and_hour(df)
        df = cyclical_encoding(df)
        df = lagged_consumption(df)
        df = rolling_window(df)
        df = lagged_trend(df)

        # We check if dataset_linear already exists
        if not output_path_linear.exists():
            df_linear = df.copy()
            df_linear = seasons_linear(df_linear)
            df_linear = interactions_linear(df_linear)

            df_linear.to_parquet(PATH_SAVE_LINEAR)
            print(f"{PATH_SAVE_LINEAR} has been successfully saved")
        else :
            df_linear = pd.read_parquet(output_path_linear)
            print(f"{output_path_linear} already exist")

        datasets[0] = df_linear
        
        # We check if dataset_tree already exists
        if not output_path_tree.exists():
            df_tree = df.copy()
            df_tree = seasons_tree(df_tree)
            df_tree = interactions_tree(df_tree)

            df_tree.to_parquet(PATH_SAVE_TREE)
            print(f"{PATH_SAVE_TREE} has been successfully saved")

        else :
            df_tree = pd.read_parquet(output_path_tree)
            print(f"{output_path_tree} already exist")
        
        datasets[1] = df_tree

    return datasets

    


        

    
    