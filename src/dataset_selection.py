import pandas as pd
from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet
)

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor
)
import xgboost as xgb


def select_datasets(PATH_DIR_LINEAR, PATH_DIR_TREE):
    linear_dir = PATH(PATH_DIR_LINEAR)
    tree_dir = PATH(PATH_DIR_TREE)

    lr = LinearRegression()
    lso = Lasso()
    rdg = Ridge()
    eln = ElasticNet()
    models = [lr, lso, rdg, eln]
    
    for path_f in linear_dir:
        df = pd.read_parquet(path_f)
        for model in models:
            
        
    
    
    


