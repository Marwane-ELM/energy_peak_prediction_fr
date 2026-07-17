"""
In this py script we train in depth the 2 best models selected previously in the pipeline (one linear model and one tree based models)
"""


import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

import src.feature_engineering as fe



def launch_training(experiment_name, PATH_DATASET_LINEAR, PATH_DATASET_TREE, best_linear_model, best_tree_based_model, horizons):
    linear_model_name = type(best_linear_model).__name__
    tree_based_model_name = type(best_tree_based_model).__name__
    
    linear_params = {
        f"{linear_model_name}__alpha" : np.arange(0.1, 2, 0.3),
        f"{linear_model_name}__tol" : [1e-3]
    }

    tree_params = {
        "loss" : ["squared_error"],
        "learning_rate" : [0.01],
        "max_iter" : [100],
        "max_leaf_nodes" : [32, 42],
        "max_depth" : [None, 10, 30, 50],
        "l2_regularization": [0.0, 0.5, 1.0]
    }

    if horizons is None:
        horizons = [0]
    
    train_model(PATH_DATASET_LINEAR, experiment_name, best_linear_model, horizons, linear_params)
    train_model(PATH_DATASET_TREE, experiment_name, best_tree_based_model, horizons, tree_params)
    

def train_model(PATH_DATASET, experiment_name, model, horizons, param_grid):
    path_f = Path(PATH_DATASET)
    dataset_name = path_f.stem


    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)

    # Columns to exclude for the standardisation for linear models
    cols_to_exclude = ['Zone_A', 'Zone_B', 'Zone_C',
       'Vacances de la Toussaint', 'Vacances de Noël', "Vacances d'Hiver",
       'Vacances de Printemps', "Vacances d'Été", 'public_holidays', 'year', 'month', 'hour', 'day_of_week', 'is_weekend']

    #cols_to_shift = ['Consommation']  #, 'lagged_1', 'lagged_2', 'lagged_48', 'lagged_336']
    #cols_to_recalculate = ["rolling_mean_24h", "rolling_std_24h", "rolling_mean_7d", "rolling_std_7d", "rolling_max_24h", "rolling_min_24h", "consumption_diff_1", "consumption_diff_48", "consumption_pct_change_1", "consumption_pct_change_48"]

    original_df = pd.read_parquet(path_f)

    for h in horizons:

        df = original_df.copy()
        # We shift the columns that need it
        df.loc[:, "Consommation"] = df["Consommation"].shift(-h)
        df = df.dropna()

        last_year = df['year'].iloc[-1]
        train_set = df[df['year'] < last_year].copy()
        test_set = df[df['year'] == last_year].copy()
        
        Y_train = train_set["Consommation"]
        Y_test = test_set['Consommation']
        X_train = train_set.drop(["Consommation", "year"], axis=1)
        X_test  = test_set.drop(["Consommation", "year"], axis=1)
        

        estimator = model
        model_name = type(estimator).__name__

        if model_name in {'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'}:
            all_cols = X_train.columns.tolist()
            cols_to_scale  = [col for col in all_cols if col not in cols_to_exclude]
            
            preprocessor = ColumnTransformer(
                transformers=[('scaler', StandardScaler(), cols_to_scale)], 
                remainder='passthrough'
            )
            
            estimator = Pipeline([
                ('preprocessor', preprocessor),
                (f'{model_name}', model)
            ])

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs = 11
        )
                    
        grid.fit(X_train, Y_train)
        best_estimator = grid.best_estimator_
        cv_rmse = -grid.best_score_
        
        y_pred = best_estimator.predict(X_test)

        #Scores on the test set
        rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
        r2 = r2_score(Y_test, y_pred)
        mae = mean_absolute_error(Y_test, y_pred)
        mse = mean_squared_error(Y_test, y_pred)
        params = grid.best_params_

        run_name = f"{model_name}_{dataset_name}_horizon_{h}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            
            mlflow.log_param("cv for grid search", 5)
            mlflow.log_param("scoring for grid search cv", "neg_root_mean_squared_error")

            mlflow.log_metric('CV root mean squared error', cv_rmse)
            mlflow.log_metric('root mean squared error', rmse)
            mlflow.log_metric('r2 score', r2)
            mlflow.log_metric('mean absolute error', mae)
            mlflow.log_metric('mean squared error', mse)
            mlflow.sklearn.log_model(best_estimator, "model")
        
            mlflow.set_tag("dataset", dataset_name)
            mlflow.set_tag("horizon", h)
            mlflow.set_tag("model", model_name)            

    print(f"{experiment_name} successfully saved")



def dump_model_from_mlflow(experiment_name, PATH_SAVE_FINAL_MODEL):
    """
    This function saves in PATH_SAVE_FINAL_MODEL all the models which perform the best on the given time perdiod horizons (5 hours horizon for example). It also returns the dictionary and the model's name shared by all the models inside of the dictionary.
    """
    mlflow.set_tracking_uri('http://localhost:5000')
    experiment = mlflow.get_experiment_by_name(f"{experiment_name}")
    runs = mlflow.search_runs(
        experiment_ids = [experiment.experiment_id],
        order_by = ['metrics.root_mean_squared_error ASC'],
        max_results=None
    )

    # The 2 best estimators from the previous step in the pipeline (the 2 best among all trained models)
    best_models = runs.sort_values(by="metrics.mean squared error")
    
    estimator1 = best_models["tags.model"].unique()[0]
    estimator2 = best_models["tags.model"].unique()[1]

    dataset_name1 = best_models[best_models["tags.model"] == estimator1]["tags.dataset"].iloc[0]
    dataset_name2 = best_models[best_models["tags.model"] == estimator2]["tags.dataset"].iloc[0]

    h = 9 
    query1 = best_models[
        (best_models["tags.model"] == estimator1) &
        (best_models["tags.dataset"] == dataset_name1) &
        (best_models["tags.horizon"].astype(str) == str(h))
    ]
    assert len(query1) > 0, f"Aucun run trouvé pour {estimator1}, {dataset_name1}, horizon={h}"
    
    query2 = best_models[
        (best_models["tags.model"] == estimator2) &
        (best_models["tags.dataset"] == dataset_name2) &
        (best_models["tags.horizon"].astype(str) == str(h))
    ]
    assert len(query2) > 0, f"Aucun run trouvé pour {estimator1}, {dataset_name1}, horizon={h}"

    score1 = query1["metrics.mean squared error"].iloc[0]
    score2 = query2["metrics.mean squared error"].iloc[0]

    best_estimator = None
    if score1 < score2 : 
        best_estimator = query1["tags.model"].unique()[0]
    else : 
        best_estimator = query2["tags.model"].unique()[0]

    
    best_runs = best_models[best_models["tags.model"] == best_estimator]
    
    all_models = {}
    for id_model in best_runs["run_id"]:
        model_uri = f"runs:/{id_model}/model"
        key = best_runs[best_runs["run_id"] == id_model]["tags.horizon"].iloc[0]
        all_models[f"{best_estimator}_{key}"] = mlflow.sklearn.load_model(model_uri)

    currente_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = Path(PATH_SAVE_FINAL_MODEL) / f"{best_estimator}_{currente_date}"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_file_path = model_dir / f"{best_estimator}_models.joblib"

    if not artifact_file_path.exists():
        joblib.dump(all_models, artifact_file_path)    
        print(f"{best_estimator} models have been successfully saved in {model_dir}")
    else : 
        print(f"{best_estimator} models already exist in {model_dir}")
    
    return best_estimator, all_models