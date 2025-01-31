import pandas as pd
import polars as pl
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import random
import pickle
from sklearn.model_selection import KFold, ShuffleSplit

def set_random_seeds(seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Before creating and training your model, call the function
set_random_seeds(42)

def load_data(path, start_dt, end_dt):
    data = pl.scan_parquet(path).filter(
        pl.col("date_id").gt(start_dt),
        pl.col("date_id").le(end_dt),
    ).collect().fill_null(0).fill_nan(0)

    return data


is_linux = False
is_win = True
is_colab = False
if is_linux:
  path = f"/home/zt/pyProjects/JaneSt/Team/data/CustomData_1_RespLags/training"
  snapshot_directory = "/home/zt/pyProjects/JaneSt/Team/scripts/George/8_0_ML_XGB/models"

elif is_win:
  path = f"E:\Python_Projects\JS2024\GITHUB_C\data\CustomData_1_RespLags\\trainData"
  snapshot_directory = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\8_0_ML_XGB\models_only_lagresp\\'

elif is_colab:
  path = f"/content/drive/MyDrive/JaneSt/CustomData3_RespLag_onDate"
  snapshot_directory = '/content/drive/MyDrive/JaneSt/8_0_ML_XGB/models'


def r2_xgb(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2

def get_model(seed):
    # XGBoost parameters
    XGB_Params = {
        'booster':          'gbtree',
        'learning_rate':        0.05,
        'n_estimators':          200,
        'early_stopping_rounds': None,
        'enable_categorical':    False,
        'grow_policy':           None,
        'max_cat_to_onehot':     None,
        'max_depth':              6,
        'subsample':             0.6,
        'colsample_bytree':      0.8,
        'reg_alpha':               2,
        'reg_lambda':              8,
        'random_state':         seed,
        'tree_method':        'hist',
        'device':             'cuda',
        'eval_metric':          r2_xgb,
        'verbosity':               1
    }
    XGB_Model = XGBRegressor(**XGB_Params)
    return XGB_Model

feature_names = [f"feature_{i:02d}" for i in range(79)]
lag_col = [f"responder_{idx}_lag_1" for idx in range(9)]
other = ['symbol_id']
label_name = 'responder_6'
weight_name = 'weight'

col_to_train = feature_names + lag_col


data = load_data(path, start_dt=1600, end_dt=1699)
X = data[col_to_train]
y = data[label_name]
w = data[weight_name]


valid_data = load_data(path, start_dt=1500, end_dt=1699)
X_val = data[col_to_train]
y_val = data[label_name]
w_val = data[weight_name]

ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for fold, (train_index, valid_index) in enumerate(ss.split(X)):
    print(f"Start Training Fold {fold + 1}")
    X_train, X_valid = X[train_index].to_numpy(), X[valid_index].to_numpy()
    y_train, y_valid = y[train_index].to_numpy(), y[valid_index].to_numpy()
    w_train, w_valid = w[train_index].to_numpy(), w[valid_index].to_numpy()



    model = get_model(42)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        sample_weight_eval_set=[w_valid],
        verbose=10  # print logs every 100 rounds
    )

    # Predict and evaluate the model here if needed
    y_pred = model.predict(X_val)
    score = r2_score(y_val, y_pred, sample_weight=w_val)
    print(f"Fold {fold + 1} R2 Score: {score}")

    # Path of the model file
    model_file_path = snapshot_directory + f'xgb_cv_{fold + 1}.json'

    # Save the model
    model.save_model(model_file_path)

    print(f"Model for fold {fold + 1} saved to {model_file_path}")

    del X_train, X_valid, y_train, y_valid, w_train, w_valid, model
    
    
