from pathlib import Path
import requests
import numpy as np
import pandas as pd
from datetime import datetime

import src.raw_preprocessing as rp
import src.feature_engineering as fe


def check_path_existence_or_create(path):
    """
    The path has to be in an Path format from Pathlib library : Path(dir1/dir2)
    """
    if not path.exists():
        print(f"{path} doesn't exists, let's create it")
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists(), f"An issue occured with the creation of the path : {path}"
        print(f"The path {path} has been successfully created\n")




path_to_data = Path("../data") #../data
path_to_artifacts = Path("../artifacts") #../artifacts
path_to_conso = path_to_data / "conso"
path_to_clean_conso = path_to_conso / "clean_conso"
path_to_real_time_conso = path_to_conso / "real_time_conso"
path_to_calendar = path_to_data / "calendar"
path_to_py_artifacts = path_to_artifacts / "py_artifacts"
path_to_weather = path_to_data / "weather"
path_to_clean_weather = path_to_weather / "clean_weather"
path_to_final_datasets = path_to_data / "final_datasets"
path_to_datasets_versions = path_to_final_datasets / "datasets_versions"
path_to_datasets_linear_models = path_to_final_datasets / "datasets_linear_models"
path_to_datasets_tree_based_models = path_to_final_datasets / "datasets_tree_based_models"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()

        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def download_data(station_population):
    """
    This function downloads the datasets required for the estimates.
    - Download the historical electricity consumption data of the past 4 years
    - Download the calendar from the government's website that contains our year intervals
    - Download the public holidays from the government's website that contains our year intervals
    - Download the historiacl weather data for the 5 main departments that contains our year intervals
    
    - Make sure to download them in the correct files
    """

    # ------------Downloading of the conso datasets --------------#
    current_year = int(datetime.now().year) - 2
    
    for i in range(0, 4):
        date = current_year - i
        url_conso = f"https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_{date}.zip"
        path = path_to_conso / f"conso_energie_{date}.zip"
        if not path.exists():
            download_file(url_conso, path)
            assert path.exists(), f"conso_energie_{date}.zip doesn't exists, an error occured during the downloading."
            print(f"conso_energie_{date}.zip has been successfully saved")
        else :
            print(f"conso_energie_{date}.zip already exists")

    # ------------Downloading of the calendar datasets --------------#
    
    url_calendar = "https://object.files.data.gouv.fr/hydra-parquet/hydra-parquet/9957d723-346e-4317-8cb3-293c94e19b2d.parquet"
    path = path_to_calendar / "calendrier_gouv.parquet"
    if not path.exists():
        download_file(url_calendar, path)
        assert path.exists(), "calendrier_gouv.parquet doesn't exists, an error occured during the downloading."
        print("calendrier_gouv.parquet has been successfully saved")
    else :
        print("calendrier_gouv.parquet already exists")
    

    # ------------Downloading of the public holidays datasets --------------#
    
    url_public_hday = "https://www.data.gouv.fr/api/1/datasets/r/6637991e-c4d8-4cd6-854e-ce33c5ab49d5"
    path = path_to_calendar / "jours_feries_metropole.csv"
    if not path.exists():
        download_file(url_public_hday, path)
        assert path.exists(), "jours_feries_metropole.csv doesn't exists, an error occured during the downloading."
        print("jours_feries_metropole.csv has been successfully saved")
    else :
        print("jours_feries_metropole.csv already exists")


    end_year = int(datetime.now().year) - 2
    start_year = end_year - 4

    for s, _ in station_population.items():
        url_weather = f"https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR/H_{s}_previous-{start_year}-{end_year}.csv.gz"
        path = path_to_weather / f"H_{s}_previous-{start_year}-{end_year}.csv.gz"
        if not path.exists():
            download_file(url_weather, path)
            assert path.exists(), f"H_{s}_previous-{start_year}-{end_year}.csv.gz doesn't exists, an error occured during the downloading."
            print(f"H_{s}_previous-{start_year}-{end_year}.csv.gz has been successfully saved")
        else :
            print(f"H_{s}_previous-{start_year}-{end_year}.csv.gz already exists")



def download_monthly_data():
    url = "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip"
    current_year = datetime.now().year
    path = path_to_real_time_conso / f"conso_energie_{current_year}.zip"
    if path.exists():
        print(f"conso_energie_{current_year}.zip already exists")
    else :
        print(f"the download of conso_energie_{current_year}.zip has started")
        download_file(url, path)
        


def final_dataset():
    input_nb_versions = str(input("Do you want to create 3 dataset versions (type 'yes' or 'no')) : ")).lower()
    num_version = None
    while input_nb_versions not in ['yes', 'no']:
        print("Incorrect answer, type 'yes' or 'no'\n")
        input_nb_versions = str(input("Do you want to create 3 dataset versions (type 'yes' or 'no')) : ")).lower()

    if input_nb_versions == 'yes':
        # If 0 then we create the 3 datasets versions
        num_version = 0
    else : 
        num_version = int(input("Select a number among 1, 2 or 3 to create the desired dataset version : "))
        while (num_version < 1) or (num_version > 3):
            print("Incorrect input number\n")
            num_version = int(input("Select a number among 1, 2 or 3 to create the desired dataset version : "))
            
        
   
    check_path_existence_or_create(path_to_data)
    check_path_existence_or_create(path_to_artifacts)
    check_path_existence_or_create(path_to_conso)
    check_path_existence_or_create(path_to_clean_conso)
    check_path_existence_or_create(path_to_real_time_conso)
    check_path_existence_or_create(path_to_calendar)
    check_path_existence_or_create(path_to_py_artifacts)
    check_path_existence_or_create(path_to_weather)
    check_path_existence_or_create(path_to_clean_weather)
    check_path_existence_or_create(path_to_final_datasets)
    check_path_existence_or_create(path_to_datasets_versions)
    check_path_existence_or_create(path_to_datasets_linear_models)
    check_path_existence_or_create(path_to_datasets_tree_based_models)

    station_population = None
    print("Downloading the required datasets")
    if num_version in [0, 1, 2]:
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
    else : 
        station_population = {
            '13': 2087658,   # Bouches-du-Rhône 
            '33': 1690493,   # Gironde           
            '44': 1487570,   # Loire-Atlantique   
            '59': 2615635,   # Nord              
            '69': 1914667,   # Rhône             
            '75': 2103778,   # Paris
        }
        
    download_data(station_population)
    
    print("\nStart of preprocessing part")

    conso = rp.preprocessing(
        path_to_clean_conso,
        path_to_conso,
        path_to_calendar,
        path_to_py_artifacts,
        path_to_weather,
        path_to_clean_weather
        
    )
    print("\nEnd of preprocessing part\n")

    print("\nStart of feature engineering part\n")
    
    if num_version == 0:
        for i in range(1, 4):
            linear_dataset, tree_dataset = fe.feature_engineering(
                conso, path_to_clean_weather, 
                path_to_datasets_versions, 
                path_to_datasets_linear_models / f"conso_v{i}_linear.parquet", 
                path_to_datasets_tree_based_models / f"conso_v{i}_tree.parquet",
                num_version = i
            )
    else : 
        linear_dataset, tree_dataset = fe.feature_engineering(
            conso, 
            path_to_clean_weather, 
            path_to_datasets_versions, 
            path_to_datasets_linear_models / f"conso_v{num_version}_linear.parquet", 
            path_to_datasets_tree_based_models / f"conso_v{num_version}_tree.parquet",
            num_version = num_version
        )

    print("\nEnd of feature engineering part\n")




if __name__ == "__main__":
    final_dataset()