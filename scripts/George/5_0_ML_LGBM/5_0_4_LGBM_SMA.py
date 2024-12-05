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


def ScoreMetric(ytrue, ypred, weight):
    """
    This function is a modification of the ready-made R-square function with sample weight.
    We have this as a column in the dataset
    """

    return r2_score(ytrue, ypred, sample_weight=weight)


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


def transf_moving_avg(data, days=10, features=['feature_12']):
    for name, sym_df in data.groupby('symbol_id'):
        data.loc[sym_df.index, features] = sym_df[features].rolling(window=days, min_periods=1).mean().astype('float32')
        
    return data

def transfrom_data(data):
    data['time_id'] = np.cos(data['time_id'])
    one_hot_encoded = pd.get_dummies(data['symbol_id'], prefix='symbol')
    data = pd.concat([data, one_hot_encoded], axis=1)
    data = data.drop('symbol_id', axis=1)
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
col_to_train = [f"symbol_{sym}" for sym in range(0,39)] + ['time_id'] + feature_names


with open(merged_scaler_df_path, 'rb') as f:
    merged_scaler_df = pickle.load(f)
log_features = ['feature_12', 'feature_13', 'feature_14', 'feature_16', 'feature_17',
                'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_71',
                'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76',
                'feature_77']



X_train = load_data(path, start_dt=1200, end_dt=1500)
X_train = X_train.fillna(0)
y_train = X_train[label_name]
w_train = X_train["weight"]
X_train = transf_moving_avg(X_train, days=10, features=log_features)
X_train = transfrom_data(X_train)
X_train = X_train[col_to_train]



X_valid = load_data(path, start_dt=1501, end_dt=1690)
X_valid = X_valid.fillna(0)
y_valid = X_valid[label_name]
w_valid = X_valid["weight"]
X_valid = transf_moving_avg(X_valid, days=10, features=log_features)
X_valid = transfrom_data(X_valid)
X_valid = X_valid[col_to_train]


X_train = X_train.fillna(0)
w_train = w_train.fillna(0)
X_train = X_train.replace([np.inf, -np.inf], 0)


X_valid = X_valid.fillna(0)
w_valid = w_valid.fillna(0)
X_valid = X_valid.replace([np.inf, -np.inf], 0)


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
    boosting_type = 'gbdt',
    n_estimators=1500,
    max_depth=11,
    learning_rate=0.01,
    colsample_bytree=0.6,
    subsample=0.6,
    random_state=42,
    reg_lambda=0.8,
    reg_alpha=0.1,
    num_leaves=800,
    verbosity=-1
)

mymetric = CustomMetricMaker(method="LGB")
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
    eval_names=["valid"], eval_metric=[mymetric.make_metric], sample_weight=w_train,
    callbacks=[log_evaluation(10),early_stopping(100, verbose=True)])




y_pred_valid = model.predict(X_valid)
valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid)
print(f"Validation R² Score: {valid_score}")


with open("E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\5_0_ML_LGBM\models\\5_0_1_base\\5_0_4_ML_lgbm_SMA.pkl", 'wb') as model_file:
    pickle.dump(model, model_file)

"""
Early stopping, best iteration is:
[515]	valid's l2: 0.627131	valid's Wgt_RSquare: 0.00877473
Validation R² Score: 0.008927443281406378


"""