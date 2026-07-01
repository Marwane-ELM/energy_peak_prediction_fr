from pathlib import Path
import numpy as np
import pandas as pd
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




path_to_data = Path(PATH_TO_DATA) #../data
path_to_artifacts = Path(PATH_ARTIFACTS) #../artifacts
path_to_conso = path_to_data / "conso"
path_to_clean_conso = path_to_conso / "clean_conso"
path_to_calendar = path_to_data / "calendar"
path_to_py_artifacts = path_to_artifacts / "py_artifacts"
path_to_weather = path_to_data / "weather"
path_to_clean_weather = path_to_weather / "clean_weather"
path_to_final_datasets = path_to_data / "final_datasets"
path_to_datasets_versions = path_to_final_datasets / "datasets_versions"
path_to_datasets_linear_models = path_to_final_datasets / "datasets_linear_models"
path_to_datasets_tree_based_models = path_to_final_datasets / "datasets_tree_based_models"

check_path_existence_or_create(path_to_data)
check_path_existence_or_create(path_to_artifacts)
check_path_existence_or_create(path_to_conso)
check_path_existence_or_create(path_to_clean_conso)
check_path_existence_or_create(path_to_calendar)
check_path_existence_or_create(path_to_py_artifacts)
check_path_existence_or_create(path_to_weather)
check_path_existence_or_create(path_to_clean_weather)
check_path_existence_or_create(path_to_final_datasets)
check_path_existence_or_create(path_to_datasets_versions)
check_path_existence_or_create(path_to_datasets_linear_models)
check_path_existence_or_create(path_to_datasets_tree_based_models)


def final_dataset():

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
    linear_dataset, tree_dataset = fe.feature_engineeiring(
        conso, path_to_clean_weather, path_to_datasets_versions, 
        path_to_datasets_linear_models, path_to_datasets_tree_based_models
    )
    print("\nEnd of feature engineering part\n")
