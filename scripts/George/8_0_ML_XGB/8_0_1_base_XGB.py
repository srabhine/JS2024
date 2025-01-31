import pandas as pd
import polars as pl
import numpy as np
import os
from sklearn.metrics import r2_score
import random
import pickle
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

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
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    merged_scaler_df_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_df.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/models/7_base_huberLoss"
    feature_dict_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"

else:
    path = f"E:\Python_Projects\JS2024\GITHUB_C\data\CustomData_1_RespLags\\trainData"
    merged_scaler_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\merged_scalers_df.pkl'
    scaler_std_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\scaler_std_df.pkl'
    feature_dict_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"
    model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\1_0_NN_PlainVanilla\model_save\model_6_perSymbol_scale"

feature_names = [f"feature_{i:02d}" for i in range(79)]
lag_col = [f"responder_{idx}_lag_1" for idx in range(9)]
other = ['symbol_id']
feature_names_mean = [f"feature_{i:02d}_mean" for i in range(79)]
feature_names_std = [f"feature_{i:02d}_std" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'

model_saving_name = "model_0_base_{epoch:02d}.keras"
col_to_train = other + feature_names + lag_col


#
# # X_train = data_train[feature_names]
# X_train = load_data(path, start_dt=500, end_dt=1659)
# y_train = X_train[label_name]
# w_train = X_train["weight"]
# X_train = X_train[col_to_train]
#
# X_valid = load_data(path, start_dt=1600, end_dt=1699)
# y_valid = X_valid[label_name]
# w_valid = X_valid["weight"]
# X_valid = X_valid[col_to_train]


# Define custom R² calculation
def r2_xgb(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return -r2

# def get_model(seed):
#     # XGBoost parameters
#     XGB_Params = {
#         'booster':          'gbtree',
#         'learning_rate':        0.01,
#         'n_estimators':         2000,
#         'early_stopping_rounds': 100,
#         'enable_categorical':   True,
#         'grow_policy':   'lossguide',
#         'max_cat_to_onehot':       4,
#         'max_depth':              14,
#         'subsample':             0.6,
#         'colsample_bytree':      0.7,
#         'reg_alpha':               2,
#         'reg_lambda':              8,
#         'random_state':         seed,
#         'tree_method':        'hist',
#         'device':             'cuda',
#         'n_gpus':                  1,
#         'eval_metric':        r2_xgb,
#         'verbosity':               1
#     }
#     XGB_Model = XGBRegressor(**XGB_Params)
#     return XGB_Model


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
        'n_gpus':                  1,
        'eval_metric':          r2_xgb,
        'verbosity':               1
    }
    XGB_Model = XGBRegressor(**XGB_Params)
    return XGB_Model


data = load_data(path, start_dt=500, end_dt=1699)
X = data[col_to_train]
y = data[label_name]
w = data[weight_name]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, valid_index) in enumerate(kf.split(X)):
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
    y_pred = model.predict(X_valid)
    score = r2_score(y_valid, y_pred, sample_weight=w_valid)
    print(f"Fold {fold + 1} R2 Score: {score}")
    
    # Path of the model file
    model_file_path = f'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\8_0_ML_XGB\models\\xgb_fold_{fold + 1}.json'
    
    # Save the model
    model.save_model(model_file_path)
    
    print(f"Model for fold {fold + 1} saved to {model_file_path}")
    del X_train, X_valid, y_train, y_valid, w_train, w_valid, model
