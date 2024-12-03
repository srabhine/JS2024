import pandas as pd
import polars as pl
import numpy as np
import os
from sklearn.metrics import r2_score
import random
import pickle
from xgboost import XGBRegressor

def set_random_seeds(seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Before creating and training your model, call the function
set_random_seeds(42)

def load_data(path, start_dt, end_dt):
    data = pl.scan_parquet(path
                           ).select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
        pl.all(),
    ).filter(
        pl.col("date_id").gt(start_dt),
        pl.col("date_id").le(end_dt),
    )

    data = data.collect().to_pandas()

    data.replace([np.inf, -np.inf], 0, inplace=True)
    data = data.fillna(0)
    return data




is_linux = False
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    merged_scaler_df_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_df.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/models/7_base_huberLoss"
    feature_dict_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"

else:
    path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    merged_scaler_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\merged_scalers_df.pkl'
    scaler_std_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\scaler_std_df.pkl'
    feature_dict_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"
    model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\1_0_NN_PlainVanilla\model_save\model_6_perSymbol_scale"

feature_names = [f"feature_{i:02d}" for i in range(79)]
feature_names_mean = [f"feature_{i:02d}_mean" for i in range(79)]
feature_names_std = [f"feature_{i:02d}_std" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'

model_saving_name = "model_0_base_{epoch:02d}.keras"
col_to_train = feature_names



# X_train = data_train[feature_names]
X_train = load_data(path, start_dt=1450, end_dt=1500)
y_train = X_train[label_name]
w_train = X_train["weight"]
X_train = X_train[col_to_train]

X_valid = load_data(path, start_dt=1650, end_dt=1690)
y_valid = X_valid[label_name]
w_valid = X_valid["weight"]
X_valid = X_valid[col_to_train]


# Define custom R² calculation
def r2_xgb(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return -r2

def get_model(seed):
    # XGBoost parameters
    XGB_Params = {
        'booster':          'gbtree',
        'learning_rate':        0.01,
        'n_estimators':         2000,
        'early_stopping_rounds': 100,
        'enable_categorical':   True,
        'grow_policy':   'lossguide',
        'max_cat_to_onehot':       4,
        'max_depth':              14,
        'subsample':             0.6,
        'colsample_bytree':      0.7,
        'reg_alpha':               2,
        'reg_lambda':              8,
        'random_state':         seed,
        'tree_method':        'hist',
        'device':             'cuda',
        'n_gpus':                  1,
        'eval_metric':        r2_xgb,
        'verbosity':               1
    }
    XGB_Model = XGBRegressor(**XGB_Params)
    return XGB_Model


model = get_model(42)
model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_valid, y_valid)],
    sample_weight_eval_set=[w_valid],
    verbose=10  # print logs every 100 rounds
)




y_pred_valid = model.predict(X_valid)
valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid)
print(f"Validation R² Score: {valid_score}")


with open("E:\Python_Projects\JS2024\GITHUB_C\scripts\George\8_0_ML_XGB\models\\8_0_1_ML_base_xgb.pkl", 'wb') as model_file:
    pickle.dump(model, model_file)

