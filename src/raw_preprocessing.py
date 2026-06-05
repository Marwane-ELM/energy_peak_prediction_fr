import numpy as np
import pandas as pd
import joblib as jb
import os.path
from datetime import datetime


# ------- Historical data about power consumption --------

def conso_preprocess(PATH_CONSO):
    """
    Input : The path to the raw consommation datasets
    Output : A new dataset composed by the merged + cleaned + preprocessed raw datasets located in PATH_CONSO/
    """

    conso = pd.DataFrame() 
    for i in range(1, 5):
        df = pd.read_csv(f"{PATH_CONSO}conso_energie_202{i}.xls", sep="\t", encoding="latin1", low_memory=False)
        conso = pd.concat([conso,df])

    conso = conso.shift(axis=1)
    conso["Périmètre"] = conso.index
    conso = conso.reset_index()
    conso = conso.drop(["index"], axis=1)
    conso = conso[conso["Consommation"].notna()]
    conso = conso[["Date", "Heures", "Consommation"]]
    conso["Date"] = conso["Date"].apply(lambda x : datetime.strptime(x, "%Y-%m-%d").date())
    conso["Heures"] = conso["Heures"].apply(lambda x : datetime.strptime(x, "%H:%M").time())

    return conso




# ------- SCHOOL HOLIDAYS --------

def is_holiday(date, zone, holiday_ranges):
    #date_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return any(start <= date < end for _, start, end in holiday_ranges[zone])


def name_holiday(x, holiday_ranges):
    date_dt = x["Date"]
    current_zone = ""
    if x["Zone_A"] == True:
        current_zone = "Zone_A"
    elif x["Zone_B"] == True:
        current_zone = "Zone_B"
    elif x["Zone_C"] == True:
        current_zone = "Zone_C"
    else : 
        return x
    for holiday_name, start, end in holiday_ranges[current_zone]:
        if start <= date_dt < end:
            x[holiday_name] = 1
            break
    return x



def school_holidays_preprocess(conso_df, PATH_HOLIDAYS, PATH_ARTIFACTS):
    """
    This function takes as input the dataset on energy consumption and 2 paths depending on where it's called from
    """

    conso = conso_df
    holidays = pd.read_excel(f"{PATH_HOLIDAYS}calendrier_gouv.xlsx")
    holidays["Zones"] = holidays["Zones"].apply(lambda d: d.replace(' ', '_'))

    holiday_names = [
    "Vacances de la Toussaint", 
    "Vacances de Noël", 
    "Vacances d'Hiver", 
    "Vacances de Printemps", 
    "Vacances d'Été"
    ]
    zones = ["Zone_A", "Zone_B", "Zone_C"]
    new_holidays = conso[["Date"]]
    
    # ---- 
    for zone in zones:    
        new_holidays[zone] = False
    for name in holiday_names:
        new_holidays[name] = 0


    # -----
    # We store the holidays for each zone in a dictionary
    holiday_ranges = {}
    # Either we load it or we create it if needed
    if not os.path.isfile(f"{PATH_ARTIFACTS}holiday_ranges.pkl"):
        for zone in zones:
            # We select the date of the new dataset to facilitate the automation of the project in the future
            min_year = new_holidays["Date"].iloc[0].year
            max_year = new_holidays["Date"].iloc[-1].year
            ranges = []
            for date in range(min_year, max_year+1):
                for holiday in holiday_names:
                    # The holidays "Vacances de Noël" start at "date-1" and end at "date".
                    if holiday == "Vacances de Noël":
                        holidays_filtered = holidays[
                                            (holidays["Zones"] == zone) & 
                                            (holidays["Description"] == holiday) &
                                            (holidays["annee_scolaire"] == f"{date-1}-{date}")
                                        ]
                    else : 
                        holidays_filtered = holidays[
                                                (holidays["Zones"] == zone) & 
                                                (holidays["Description"] == holiday) &
                                                (holidays["annee_scolaire"] == f"{date}-{date+1}")
                                            ]
                    start = datetime.strptime(holidays_filtered["Date de début"].iloc[0][:10], "%Y-%m-%d").date()
                    end = datetime.strptime(holidays_filtered["Date de fin"].iloc[0][:10], "%Y-%m-%d").date()
                    ranges.append((holiday, start, end))
            holiday_ranges[zone] = ranges
        jb.dump(holiday_ranges, f"{PATH_ARTIFACTS}holiday_ranges.pkl")
        print("(Dictionnary successfully created)")
        
    else : 
        holiday_ranges = jb.load(f"{PATH_ARTIFACTS}holiday_ranges.pkl")  
        print("(Dictionnary successfully loaded)")

    # ------
    
    for zone in zones:
        new_holidays[zone] = new_holidays["Date"].apply(lambda d : is_holiday(d, zone, holiday_ranges))
    
    new_holidays = new_holidays.apply(lambda x: name_holiday(x, holiday_ranges), axis=1)
    
    return new_holidays


# --------------------------------

def public_holidays_preprocess(conso_df, PATH_PUBLIC_HDAY):
    feries = pd.read_csv(f"{PATH_PUBLIC_HDAY}jours_feries_metropole.csv").drop(["annee", "zone"], axis=1) 
    feries["date"] = feries["date"].apply(lambda x : datetime.strptime(x, "%Y-%m-%d").date())
    feries = feries[
        (feries["date"] >= conso_df["Date"].iloc[0]) &
        (feries["date"] <= conso_df["Date"].iloc[-1])
        ]
    public_holidays = set(feries["date"])
    conso_df["public_holidays"] = pd.to_datetime(conso_df["Date"]).isin(public_holidays) + 0
    


# ----- SELECTING THE BEST WEATHER STATION 

def monitoring_nan(df):
    print("There are", len(df["NUM_POSTE"].unique()), "stations")
    
    print("Percentage of NaN per column")
    print((df.isnull().mean() * 100).round(1))
    
    print("\nPercentage of NaN per station")
    print(df.groupby('NUM_POSTE')[['T', 'U', 'FF', 'PMER', 'RR1']].apply(
        lambda x: x.isnull().mean() * 100))


def select_best_station(df, conso):    
    grid_scores = df.groupby('NUM_POSTE')[['T', 'U', 'FF', 'PMER', 'RR1']].apply(lambda x: x.isnull().mean() * 100)
    grid_scores_array = np.array(grid_scores) 
    x, y = grid_scores_array.shape
    
    weights = np.array([3, 2, 2, 1, 1]).reshape((y, 1))
    best_station_score = grid_scores_array.dot(weights)
    best_station = int(grid_scores.iloc[best_station_score.argmin()].name)

    print("The best station has a weighted average of : ", best_station_score.min())
    print("Best station : ", best_station)
    
    df = df[df["NUM_POSTE"] == best_station]
    assert len(conso) == len(df) * 2, print("The sizes of this dataset and the main dataset conso don't match together\n",
                                           f"Current dataset size : {len(df)}", f"Main dataset size : {len(conso)}")

    
    return df



# Adding the timestamps every half-hour

def half_hour(df, conso):
    timestamps = conso[["Date", "Heures"]].sort_values(["Date", "Heures"])
    
    df = timestamps.merge(df, on=["Date", "Heures"], how="left")
    df = df.sort_values(["Date", "Heures"]).reset_index(drop=True)
    df["NUM_POSTE"] = df["NUM_POSTE"].loc[df["NUM_POSTE"].first_valid_index()].astype(int)
    df["LAT"] = df["LAT"].loc[df["LAT"].first_valid_index()]
    df["LON"] = df["LON"].loc[df["LON"].first_valid_index()]
    return df
    


# FILLING THE MISING VALUES WITH THE LINEAR INTERPOLATION METHOD

def interpolation(x, n1, n2):
    x1, X1 = n1
    x2, X2 = n2
    return X1 + (x-x1)*(X2-X1)/(x2-x1)


def time_to_float(t):
    """Convert datetime.time to float hours 
    example :  14:30 -> 14.5
    """
    return t.hour + t.minute / 60


def interpolation_col(df):
    df = df.reset_index(drop=True)
    columns = list(set(df.columns) - set(["NUM_POSTE", "LAT", "LON", "Date", "Heures"]))
    
    for col in columns:
        # If the column contains Nan Values, we apply the interpolation method on it
        if (df[col].isnull().mean() * 100) != 0.0 :
            for i in range(len(df)):
                if(pd.isna(df[col].iloc[i])):
                    # If the missing value is in the first row, we choose the value of the next row
                    
                    if(i == 0):
                        df[i, col] = df[col].iloc[i+1]                        
                    # If the missing value is in the last row, we choose the value of the last row
                    elif i == len(df)-1 : 
                        df[i, col] = df[col].iloc[i-1]
                    else : 
                        
                        t = time_to_float(df["Heures"].iloc[i])
                        t1, T1 = time_to_float(df["Heures"].iloc[i-1]), df[col].iloc[i-1]    
                        # We select the index of the next valid row (Not NaN value)
                        # first_valid_index returns an index, not a position
                        idx_next = df[col].iloc[i+1:].first_valid_index()
                        if idx_next is None:
                            df.loc[i, col] = T1
                            continue

                        t2, T2 = time_to_float(df["Heures"].loc[idx_next]), df[col].loc[idx_next]
                        
                        df.loc[i, col] = interpolation(t, (t1, T1), (t2, T2))
    return df



def interpolate_pd(df):
    columns = list(set(df.columns) - set(["NUM_POSTE", "LAT", "LON", "Date", "Heures"]))
    df[columns] = df[columns].interpolate(method="linear", limit=12)
            
            
    




