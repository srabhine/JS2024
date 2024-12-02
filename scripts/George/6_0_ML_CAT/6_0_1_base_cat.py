import pandas as pd
import polars as pl
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import random


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


def save_intermediate_model(model, directory, epoch):
    """
    Save the model to the specified directory with the given epoch number.
    """
    model_path = os.path.join(directory, f"catboost_model_epoch_{epoch:03d}.cbm")
    model.save_model(model_path, format="cbm")
    print(f"Model at epoch {epoch:03d} saved.")


# Update your CatBoost callback:
class CustomIterationCallback:
    def __init__(self, save_dir, interval=100):
        self.save_dir = save_dir
        self.interval = interval
    
    def after_iteration(self, info):
        # Check if it's time to save the model
        if info.iteration % self.interval == 0:
            save_intermediate_model(info.model, self.save_dir, info.iteration)


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
X_train = load_data(path, start_dt=1200, end_dt=1500)
# X_train = data_train[feature_names]
y_train = X_train[label_name]
w_train = X_train["weight"]
X_train = X_train[col_to_train]
# del data_train

X_valid = load_data(path, start_dt=1501, end_dt=1690)
# X_valid = data_valid[feature_names]
y_valid = X_valid[label_name]
w_valid = X_valid["weight"]
X_valid = X_valid[col_to_train]
# del data_valid


# Define custom R² calculation
def r2_val(y_true, y_pred, sample_weight):
    """Calculate weighted R² value."""
    numerator = np.average((y_pred - y_true) ** 2, weights=sample_weight)
    denominator = np.average(y_true ** 2, weights=sample_weight) + 1e-38
    r2 = 1 - numerator / denominator
    return r2


def get_model(seed):
    CatBoost_Params = {
        'learning_rate': 0.01,
        'depth': 11,
        'iterations': 3000,
        'subsample': 0.6,
        'colsample_bylevel': 0.8,
        'l2_leaf_reg': 6,
        'random_seed': seed,
        'verbose': 1,  # Control the verbosity
        'loss_function': 'RMSE',
        'eval_metric': 'R2',  # Include R2 as an evaluation metric
        'snapshot_file': 'catboost_snapshot',  # File to save model state
        'snapshot_interval': 100,  # Save the model every 100 iterations
        'allow_writing_files': True,  # Allow writing to files (required for snapshots)
    }
    CatBoost_Model = CatBoostRegressor(**CatBoost_Params)
    return CatBoost_Model



snapshot_directory = 'E:/Python_Projects/JS2024/GITHUB_C/scripts/George/5_0_ML_LGBM/models/snapshots/'
os.makedirs(snapshot_directory, exist_ok=True)
model = get_model(seed=42)
model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_valid, y_valid)],
    verbose=1  # Adjust logging frequency
)



y_pred_valid = model.predict(X_valid)
valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid)
print(f"Validation R² Score: {valid_score}")




