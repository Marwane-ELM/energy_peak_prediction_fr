"""
This py script is used for the selection of the best datasets among all the datasets located 
in the folder '../data/final_datasets/datasets_linear_models or datasets_tree_based_models'
"""


import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

def log_models(PATH_DIR_LINEAR, PATH_DIR_TREE):
    linear_models = [
        LinearRegression(),
        Lasso(tol=1e-2, max_iter=5000), 
        Ridge(), 
        ElasticNet(tol=1e-2, max_iter=5000)
    ]
    tree_models   = [GradientBoostingRegressor(), HistGradientBoostingRegressor(), RandomForestRegressor()]

    log_experiment(PATH_DIR_LINEAR, 'Linear Experiment', linear_models)
    log_experiment(PATH_DIR_TREE, 'Trees Experiment', tree_models)

    

def log_experiment(PATH_DIR, experiment_name, models):
    """
    This function trains simple ML models on all the types of datasets (tailored for linear or tree based models).
    """
    data_dir = Path(PATH_DIR)

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)

    # Columns to exclude for the standardisation for linear models
    cols_to_exclude = ['Zone_A', 'Zone_B', 'Zone_C',
       'Vacances de la Toussaint', 'Vacances de Noël', "Vacances d'Hiver",
       'Vacances de Printemps', "Vacances d'Été", 'public_holidays', 'year', 'month', 'hour', 'day_of_week', 'is_weekend']



    for path_f in data_dir.glob("*.parquet"):
        dataset_name = path_f.stem
        df = pd.read_parquet(path_f)
        
        last_year = df['year'].iloc[-1]
        train_set = df[df['year'] < last_year].copy()
        test_set = df[df['year'] == last_year].copy()
        
        Y_train = train_set["Consommation"]
        Y_test = test_set['Consommation']
        X_train = train_set.drop(["Consommation", "year"], axis=1)
        X_test  = test_set.drop(["Consommation", "year"], axis=1)
        
        for model in models:
            model_name = type(model).__name__

            if model_name in {'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'}:
                all_cols = X_train.columns.tolist()
                cols_to_scale  = [col for col in all_cols if col not in cols_to_exclude]
                
                preprocessor = ColumnTransformer(
                    transformers=[('scaler', StandardScaler(), cols_to_scale)], 
                    remainder='passthrough'
                )
                
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    (f'{model_name}', model)
                ])
            
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)

            r2 = model.score(X_test, Y_test)
            mae = mean_absolute_error(Y_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            params = model.get_params()

            with mlflow.start_run(run_name=f'{model_name}_{dataset_name}'):
                mlflow.log_params(params)

                mlflow.log_metric('r2 score', r2)
                mlflow.log_metric('mean absolute error', mae)
                mlflow.log_metric('mean squared error', mse)
            
                mlflow.set_tag('Training Info', f'{model_name} for {dataset_name}')
            

    print(f"{experiment_name} successfully saved")

            