import pandas as pd
import polars as pl
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score
import random
import pickle
from lightgbm import LGBMRegressor
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
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

class CustomMetricMaker:
    "This class makes the custom metric for LGBM and XGBoost early stopping"

    def __init__(self, method):
        self.method = method

    def make_metric(self, ytrue, ypred, weight):
        """
        This method returns the relevant metric for LGBM and XGB.
        Catboost has a slightly different signature for the same- will be provided in version 2
        """

        if "LGB" in self.method:
            return 'Wgt_RSquare', ScoreMetric(ytrue, ypred, weight), True
        else:
            return ScoreMetric(ytrue, ypred, weight)


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
def r2_val(y_true, y_pred, sample_weight):
    """Calculate weighted R² value."""
    numerator = np.average((y_pred - y_true) ** 2, weights=sample_weight)
    denominator = np.average(y_true ** 2, weights=sample_weight) + 1e-38
    r2 = 1 - numerator / denominator
    return r2

model = LGBMRegressor(
    device="gpu",
    objective="regression_l2",
    n_estimators=1500,
    max_depth=11,
    learning_rate=0.01,
    colsample_bytree=0.6,
    subsample=0.6,
    random_state=42,
    reg_lambda=0.8,
    reg_alpha=0.1,
    num_leaves=800,
    verbosity=1
)

mymetric = CustomMetricMaker(method="LGB")
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
    eval_names=["Test"], eval_metric=[mymetric.make_metric], sample_weight=w_train,
    callbacks=[log_evaluation(100),early_stopping(100, verbose=True)])




y_pred_valid = model.predict(X_valid)
valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid)
print(f"Validation R² Score: {valid_score}")


with open("E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\5_0_ML_LGBM\models\\5_0_1_base/5_0_1_ML_base_lgbm.pkl", 'wb') as model_file:
    pickle.dump(model, model_file)

