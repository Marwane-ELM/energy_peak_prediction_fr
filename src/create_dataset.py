from pathlib import Path
import numpy as np
import pandas as pd
import os.path


def final_dataset():

    """
    This function download, clean and merge all the required datasets for the project
    """
    conso = None
    if os.path.isfile("data/conso/conso.parquet"):
        conso = pd.read_parquet("data/conso/conso.parquet")
        print("conso dataset succesfully loaded")
    else : 
        conso = rp.conso_preprocess("data/conso/")
        holidays2 = rp.school_holidays_preprocess(conso, PATH_HOLIDAYS="data/calendar/", PATH_ARTIFACTS="artifacts/py_artifacts/")
        conso = pd.concat([conso, holidays2.drop("Date", axis=1)], axis=1)
        rp.public_holidays_preprocess(conso, "data/calendar/")
        conso.to_parquet("data/conso/clean_conso/conso.parquet")
        print("conso dataset succesfully created")

    return None