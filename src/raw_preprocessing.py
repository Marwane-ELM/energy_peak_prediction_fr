import numpy as np
import pandas as pd
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
    return conso




# ------- SCHOOL HOLIDAYS --------

def is_holiday(date_str, zone, holiday_ranges):
    date_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    return any(start <= date_dt < end for _, start, end in holiday_ranges[zone])


def name_holiday(x, holiday_ranges):
    date_dt = datetime.strptime(x["Date"][:10], "%Y-%m-%d")
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



def school_holidays_preprocess(conso_df, PATH_HOLIDAYS):
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
    for zone in zones:
        # We select the date of the new dataset to facilitate the automation of the project in the future
        min_year = int(new_holidays["Date"].iloc[0][:4])
        max_year = int(new_holidays["Date"].iloc[-1][:4])
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
                start = datetime.strptime(holidays_filtered["Date de début"].iloc[0][:10], "%Y-%m-%d")
                end = datetime.strptime(holidays_filtered["Date de fin"].iloc[0][:10], "%Y-%m-%d")
                ranges.append((holiday, start, end))
        holiday_ranges[zone] = ranges

    # ----
    
    for zone in zones:
        new_holidays[zone] = new_holidays["Date"].apply(lambda d : is_holiday(d, zone, holiday_ranges))
    
    new_holidays = new_holidays.apply(lambda x: name_holiday(x, holiday_ranges), axis=1)

    return new_holidays
        


