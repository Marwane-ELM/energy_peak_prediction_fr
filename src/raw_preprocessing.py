from pathlib import Path
import numpy as np
import pandas as pd
import joblib as jb
import os.path
import re
from datetime import datetime


# ------- Historical data about power consumption --------

def conso_preprocess(PATH_CONSO):
    """
    Input : The path to the raw energy consumption datasets
    Output : A new dataset composed by the merged + cleaned + preprocessed raw datasets located in PATH_CONSO/
    """

    conso = pd.DataFrame() 
    for i in range(1, 5):
        df = pd.read_csv(PATH_CONSO / f"conso_energie_202{i}.zip", compression='zip', sep="\t", encoding="latin1", low_memory=False)
        conso = pd.concat([conso,df])

    conso = conso.shift(axis=1)
    conso["Périmètre"] = conso.index
    #conso = conso.drop(["index"], axis=1)
    conso = conso[conso["Consommation"].notna()]
    conso = conso[["Date", "Heures", "Consommation"]]
    conso["Date"] = conso["Date"].apply(lambda x : datetime.strptime(x, "%Y-%m-%d").date())
    conso["Heures"] = conso["Heures"].apply(lambda x : datetime.strptime(x, "%H:%M").time())
    conso = conso.reset_index(drop=True)


    return conso



# ------- SCHOOL HOLIDAYS --------

def is_holiday(date, zone, holiday_ranges):
    return any(start <= date < end for _, start, end in holiday_ranges[zone]) + 0


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

    holidays = pd.read_parquet(PATH_HOLIDAYS / "calendrier_gouv.parquet")
    holidays["Zones"] = holidays["Zones"].apply(lambda d: d.replace(' ', '_'))

    holiday_names = [
    "Vacances de la Toussaint", 
    "Vacances de Noël", 
    "Vacances d'Hiver", 
    "Vacances de Printemps", 
    "Vacances d'Été"
    ]
    zones = ["Zone_A", "Zone_B", "Zone_C"]
    new_holidays = conso_df[["Date"]]
    
    # ---- 
    for zone in zones:    
        new_holidays[zone] = False
    for name in holiday_names:
        new_holidays[name] = 0


    # -----
    # We store the holidays for each zone in a dictionary
    holiday_ranges = {}
    # Either we load it or we create it if needed
    if not os.path.isfile(PATH_ARTIFACTS / "holiday_ranges.pkl"):
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
                    start = pd.to_datetime(holidays_filtered["Date de début"].iloc[0]).date()
                    end = pd.to_datetime(holidays_filtered["Date de fin"].iloc[0]).date()
                    ranges.append((holiday, start, end))
            holiday_ranges[zone] = ranges
        jb.dump(holiday_ranges, PATH_ARTIFACTS / "holiday_ranges.pkl")
        
    else : 
        holiday_ranges = jb.load(PATH_ARTIFACTS / "holiday_ranges.pkl")  

    # ------
    
    for zone in zones:
        new_holidays[zone] = new_holidays["Date"].apply(lambda d : is_holiday(d, zone, holiday_ranges))
    
    new_holidays = new_holidays.apply(lambda x: name_holiday(x, holiday_ranges), axis=1)
    conso_df = pd.concat([conso_df, new_holidays.drop("Date", axis=1)], axis=1)

    return conso_df


# --------------------------------

def public_holidays_preprocess(conso_df, PATH_PUBLIC_HDAY):
    feries = pd.read_csv(PATH_PUBLIC_HDAY / "jours_feries_metropole.csv").drop(["annee", "zone"], axis=1) 
    feries["date"] = feries["date"].apply(lambda x : datetime.strptime(x, "%Y-%m-%d").date())
    feries = feries[
        (feries["date"] >= conso_df["Date"].iloc[0]) &
        (feries["date"] <= conso_df["Date"].iloc[-1])
        ]
    public_holidays = set(feries["date"])
    conso_df["public_holidays"] = pd.to_datetime(conso_df["Date"]).isin(public_holidays) + 0
    return conso_df
    


# ----- SELECTING THE BEST WEATHER STATION 

def monitoring_nan(df):
    print("Nb of stations : ", len(df["NUM_POSTE"].unique()))
    
    print("Percentage of NaN per column :")
    print((df.isnull().mean() * 100).round(1))
    
    print("\nPercentage of NaN per station :")
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
    
    df = df[df["NUM_POSTE"] == best_station].drop_duplicates(subset=["Date", "Heures"])
    assert len(conso) == len(df) * 2, print("The sizes of this dataset and the main dataset conso don't match together\n", f"Current dataset size*2 : {len(df) * 2}", f"Main dataset size : {len(conso)}")

    
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
    df = df.copy()
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
    # The limit_direction is very important for our problem
    df[["T", "U"]] = df[["T", "U"]].interpolate(method="linear", limit=12, limit_direction="both")  
    df[["PMER"]] = df[["PMER"]].interpolate(method="linear", limit=24, limit_direction="both")  
    df['RR1'] = df['RR1'].fillna(0.0)

    # After interpolation we fill any NaN with the column median
    for col in ['T', 'U', 'FF', 'PMER']:
        df[col] = df[col].fillna(df[col].median())
    return df
    
    

def weather_clean_all(conso, PATH_WEATHER_FILES, PATH_SAVE_WEATHER_FILES):
    """
    This function preprocesses all the weather files (.gz) from PATH_WEATHER_FILES and saves them
    in PATH_SAVE_WEATHER_FILES in .parquet format (to save space)
    """
    data_dir = Path(PATH_WEATHER_FILES)
    cols_to_keep = [
        'NUM_POSTE',        # identifies the city
        'LAT', 'LON',       # coordinates: used to query Open-Meteo
        'AAAAMMJJHH',       # timestamp
        # Usefull features that we can found with the API of open-meteo
        'T',                # temperature_2m        
        'U',                # relative_humidity_2m  
        'FF',               # wind_speed_10m        
        'PMER',             # pressure_msl          
        'RR1',              # precipitation         
    ]
    min_date = conso["Date"].iloc[0]
    max_date = conso["Date"].iloc[-1]


    for path_f in data_dir.glob("*.gz"):
        output_path = Path(PATH_SAVE_WEATHER_FILES) / path_f.with_suffix("").with_suffix(".parquet").name
        num = re.findall(r'\d+', os.path.basename(path_f))[0]

        if not output_path.exists():
            print("Loading the file of the department n°", num, "...")
            # We load the dataset and keep the essential columns
            df = pd.read_csv(path_f, sep =';', compression="gzip")[cols_to_keep] 
            print("File loaded with success")
    
            print("Adding the Date and Hours")
            df["Date"] = df["AAAAMMJJHH"].apply(lambda x : str(str(x)[:8]))
            df["Date"] = df["Date"].apply(lambda x : datetime.strptime(x, '%Y%m%d').date())
            
            df["Heures"] = df["AAAAMMJJHH"].apply(lambda x : str(str(x)[8:]) + ':00')
            df["Heures"] = df["Heures"].apply(lambda x : datetime.strptime(x, '%H:%M').time())
            
            df = df.drop("AAAAMMJJHH", axis=1)
            assert len(df["Heures"].unique()) == 24, "There are missing time frames"
    
            
            df = df[
                (df["Date"]>= min_date) & 
                (df["Date"]<= max_date)
                ]
            
            print(f"Before selecting the best station for department n°{num}\n")
            monitoring_nan(df)
            df = select_best_station(df, conso)
            print(f"\nAfter selecting the best station for departement n°{num}\n")
            monitoring_nan(df)
    
            print("Adding the half-hour time units\n")
            df = half_hour(df, conso)
    
            print("Interpolation of the columns")
            df = interpolate_pd(df)
    
            print("After interpolation : \n")
            monitoring_nan(df)
    
            df.to_parquet(output_path, index=False)
            
            print(f"Dataset {os.path.basename(path_f)} was cleaned and saved with success")
            print("# -------------------------------------------------------------------#\n")
        else : 
            print(f"The file for the department n°{num} already exists")

    print(f"\n\nEverything has been successfully saved in {PATH_SAVE_WEATHER_FILES}")
    
    

def dataset_v1(conso, PATH_CLEAN_WEATHER_FILES, PATH_DATASETS_VERSIONS):
    """
    This function merges the energy consumption dataset with the existing weather datasets. It keeps the temperature columns of each French department and calculates the population-weighted average for the other columns.
    12 main departments + weighted average
    """
    output_path = Path(PATH_DATASETS_VERSIONS) / "conso_v1.parquet"
    if output_path.exists():
        print(f"The file {output_path} already exists")
        return pd.read_parquet(output_path)

    data_dir = Path(PATH_CLEAN_WEATHER_FILES)
    station_population = {
        '13': 2087658,   # Bouches-du-Rhône 
        '21': 540100,   # Cote d'Or        
        '31': 1471468,   # Haute-Garonne    
        '33': 1690493,   # Gironde           
        '35': 1120666,   # Ille-et-Vilaine   
        '44': 1487570,   # Loire-Atlantique  
        '45': 691268,    # Loiret            
        '59': 2615635,  # Nord              
        '67': 1163810,  # Bas-Rhin          
        '69': 1914667,   # Rhône             
        '75': 2103778,   # Paris
        '76': 1260964,   # Seine-Maritime    
    }

    # We normalize them in order to keep the same weather units
    total_pop = sum(station_population.values())
    weights = {city: pop / total_pop for city, pop in station_population.items()}
    
    cols = ['T', 'U', 'FF', 'PMER', 'RR1']
    test = np.zeros(shape = (conso.shape[0], len(cols)))
    df_temp = pd.DataFrame(test, columns = cols)

    for path_f in data_dir.glob("*.parquet"):

        #We store the poupalation-weighted columns
        df = pd.read_parquet(path_f)
        prefix = re.findall(r'\d+', os.path.basename(path_f))[0]

        population = weights[prefix]
        df_temp += (df[cols].values * population)

        # We add the temperature columns
        df = df.add_prefix(str(prefix))  # We add a prefix to each column to recognize them    
        conso = pd.concat([conso, df[['T']].rename(columns={'T': f'{prefix}T'})], axis=1)  
        
    conso.to_parquet(output_path)
    print(f"The dataset {path_to_datasets_versions} has been successfully saved")
        
    return conso
    
    
        
        
def dataset_v2(conso, PATH_CLEAN_WEATHER_FILES, PATH_DATASETS_VERSIONS):
    """
    Only weighted averages
    """
    output_path = Path(PATH_DATASETS_VERSIONS) / "conso_v2.parquet"
    if output_path.exists():
        print(f"The file {output_path} already exists")
        return pd.read_parquet(output_path)


    data_dir = Path(PATH_CLEAN_WEATHER_FILES)
    station_population = {
        '13': 2087658,   # Bouches-du-Rhône 
        '21': 540100,   # Cote d'Or        
        '31': 1471468,   # Haute-Garonne    
        '33': 1690493,   # Gironde           
        '35': 1120666,   # Ille-et-Vilaine   
        '44': 1487570,   # Loire-Atlantique  
        '45': 691268,    # Loiret            
        '59': 2615635,  # Nord              
        '67': 1163810,  # Bas-Rhin          
        '69': 1914667,   # Rhône             
        '75': 2103778,   # Paris
        '76': 1260964,   # Seine-Maritime    
    }

    # We normalize them in order to keep the same units
    total_pop = sum(station_population.values())
    weights = {city: pop / total_pop for city, pop in station_population.items()}
    
    cols = ['T', 'U', 'FF', 'PMER', 'RR1']
    test = np.zeros(shape = (conso.shape[0], len(cols)))
    df_temp = pd.DataFrame(test, columns = cols)
    
    for path_f in data_dir.glob("*.parquet"):

        df = pd.read_parquet(path_f)
        prefix = re.findall(r'\d+', os.path.basename(path_f))[0]
        population = weights[prefix]
        df_temp += df[cols].values * population

    conso = pd.concat([conso, df_temp], axis=1)
    conso.to_parquet(output_path)
    print(f"The dataset {path_to_datasets_versions} has been successfully saved")
        
    return conso



def dataset_v3(conso, PATH_CLEAN_WEATHER_FILES, PATH_DATASETS_VERSIONS):
    """
    5 main departments + weighted averages
    """

    output_path = Path(PATH_DATASETS_VERSIONS) / "conso_v3.parquet"
    
    # If file exists, load and return it directly
    if output_path.exists():
        print(f"The file {output_path} already exists")
        return pd.read_parquet(output_path)  # ← was returning conso without weather cols

    data_dir = Path(PATH_CLEAN_WEATHER_FILES)
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
        np.zeros(shape=(conso.shape[0], len(cols))),
        columns=cols
    )

    for path_f in data_dir.glob("*.parquet"):
        df = pd.read_parquet(path_f)
        prefix = re.findall(r'\d+', os.path.basename(path_f))[0]
        
        if prefix in station_population:
            population = weights[prefix]
            df_temp += df[cols].values * population
            conso = pd.concat([conso, df[['T']].rename(columns={'T': f'{prefix}T'})], axis=1)

    
    conso = pd.concat([conso, df_temp], axis=1)
    conso.to_parquet(output_path)
    print(f"The dataset {PATH_DATASETS_VERSIONS} has been successfully saved")
    
    return conso




# ------------------------------  AUTOMATION OF THE PREPROCESSING --------------------------


def preprocessing(
    path_to_clean_conso,
    path_to_conso,
    path_to_calendar,
    path_to_py_artifacts,
    path_to_weather,
    path_to_clean_weather
    
):
    
    conso = None
    if os.path.isfile(path_to_clean_conso / "conso.parquet"):
        conso = pd.read_parquet(path_to_clean_conso / "conso.parquet")
        print("conso dataset succesfully loaded")
    else : 
        
        conso = conso_preprocess(path_to_conso)
        conso = school_holidays_preprocess(
            conso, 
            PATH_HOLIDAYS = path_to_calendar, 
            PATH_ARTIFACTS = path_to_py_artifacts
        )
        conso = public_holidays_preprocess(conso, PATH_PUBLIC_HDAY = path_to_calendar)
        conso.to_parquet(path_to_clean_conso / "conso.parquet")
        print("conso dataset succesfully created")

    
    weather_clean_all(
        conso, 
        PATH_WEATHER_FILES = path_to_weather, 
        PATH_SAVE_WEATHER_FILES = path_to_clean_weather
    )

    print("\n Preprocessing completed successfully")

    return conso
    
        
