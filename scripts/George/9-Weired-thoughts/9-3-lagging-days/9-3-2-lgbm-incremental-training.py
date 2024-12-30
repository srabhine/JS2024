import pickle
import os
import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import numpy as np
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
import random
from tqdm import tqdm
from sklearn.metrics import r2_score


with open('/home/zt/pyProjects/JaneSt/Team/scripts/George/9-Weired-thoughts/9-3-lagging-days/models/lgbm_model.pkl',
          'rb') as file:
    model = pickle.load(file)
class CONFIG:
    debug = False
    seed = 42
    target_col = "responder_6"
    lag_cols_rename = {f"responder_{idx}_lag_1": f"responder_{idx}" for idx in range(9)}
    lag_target_cols_name = [f"responder_{idx}" for idx in range(9)]
    lag_cols_original = ["date_id", "time_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
    feature_names = [f"feature_{i:02d}" for i in range(79)]
    time_features = ["sin_time_id", "cos_time_id", "sin_time_id_halfday", "cos_time_id_halfday"]
    lag_cols_post = time_features + feature_names + [f"responder_{idx}_mean" for idx in range(9)] + [
        f"responder_{idx}_std" for idx in range(9)] + [f"responder_{idx}_max" for idx in range(9)] \
                    + [f"responder_{idx}_last" for idx in range(9)]
    model_path = "/kaggle/input/janestreet-public-model/xgb_001.pkl"
    lag_ndays = 1
def create_agg_list(day, columns):
    agg_mean_list = [pl.col(c).mean().name.suffix(f"_mean") for c in columns]
    agg_std_list = [pl.col(c).std().name.suffix(f"_std") for c in columns]
    agg_max_list = [pl.col(c).max().name.suffix(f"_max") for c in columns]
    agg_last_list = [pl.col(c).last().name.suffix(f"_last") for c in columns]
    agg_list = agg_mean_list + agg_std_list + agg_max_list + agg_last_list
    return agg_list


def making_time_id_features(X_test):
    X_test['sin_time_id'] = np.sin(2 * np.pi * X_test['time_id'] / 967)
    X_test['cos_time_id'] = np.cos(2 * np.pi * X_test['time_id'] / 967)
    X_test['sin_time_id_halfday'] = np.sin(2 * np.pi * X_test['time_id'] / 483)
    X_test['cos_time_id_halfday'] = np.cos(2 * np.pi * X_test['time_id'] / 483)
    return X_test
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

def load_data(path, start_dt, end_dt):
	data = pl.scan_parquet(path).select(
			pl.all(), ).filter(
			pl.col("date_id").gt(start_dt),
			pl.col("date_id").le(end_dt),
	).fill_null(0).collect()

	return data

def make_data(path, start_dt, end_dt):
    X_train = load_data(path, start_dt=start_dt, end_dt=end_dt)
    lags = X_train.select(pl.col(CONFIG.lag_cols_original))
    lags = lags.with_columns(date_id=pl.col("date_id") + 1)
    date_ids = lags.select("date_id").unique().to_series()
    agg_list = create_agg_list(CONFIG.lag_ndays, CONFIG.lag_target_cols_name)

    result = []
    for date_id in tqdm(date_ids, total=len(date_ids)):
        try:
            # rolling N天
            #lags_ = lags.filter((pl.col("date_id") > date_id - CONFIG.lag_ndays) & (pl.col("date_id") <= date_id))
            # shift N天
            lags_ = lags.filter((pl.col("date_id") == date_id - CONFIG.lag_ndays))
            # 为了merge，将date_id统一到对应的date_id
            # 比如在统计第10天的rolling 3天的数据时, 数据中的date_id应该是8,9,10, 统一为10和主数据对应
            lags_ = lags_.with_columns(date_id=date_id)
            lags_ = lags_.group_by(["date_id", "symbol_id"], maintain_order=True).agg(agg_list)
            result.append(lags_)
        except:
            continue

    lag_Ndays = pl.concat(result).sort("date_id")
    lag_Ndays = lag_Ndays.cast({"date_id": pl.Int16})

    train = X_train.join(lag_Ndays, on=["date_id", "symbol_id"], how="left")
    X_train = train.fill_nan(0).fill_null(0).to_pandas()
    X_train = making_time_id_features(X_train)
    y_train = X_train[CONFIG.target_col]
    w_train = X_train["weight"]
    X_train = X_train[CONFIG.lag_cols_post]
    return X_train, y_train, w_train

is_linux = True
if is_linux:
    original_data_path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    training_resp_lag_path = "/home/zt/pyProjects/JaneSt/Team/data/CustomData_2_RespLags_onTime/trainData"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/1_HS_Plan/Torch-NN-models"
else:
    original_data_path = "E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    training_resp_lag_path = "E:\Python_Projects\JS2024\GITHUB_C\data\CustomData_2_RespLags_onTime\\trainData"
    model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\9-Weired-thoughts\9-2-Model-SymbSpec\models"
    model_saving_name = "model_1.kears"


X_train, y_train, w_train = make_data(original_data_path, 500, 800)
X_valid, y_valid, w_valid = make_data(original_data_path, 1601, 1689)
# Combine old and new data (example)
train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, weight=w_valid)
booster = model.booster_
# Initialize a new model with the same hyperparameters as before
params = {
    'objective': 'regression_l2',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 800,
    'max_depth': 10,
    'colsample_bytree': 0.6,
    'subsample': 0.6,
    'random_state': 42,
    'reg_lambda': 0.8,
    'reg_alpha': 0.1,
    'verbosity': -1,
    'device': 'gpu',
}

# Perform incremental training
# Note: `num_boost_round` controls how many additional boosting iterations to perform
booster = lgb.train(
    params,
    train_data,
    num_boost_round=300,  # or any other number of iterations for new data
    init_model=booster,
    valid_sets=[valid_data],
    valid_names=['validation'],
    callbacks=[log_evaluation(10),early_stopping(100, verbose=True)]
)
# Save updated model
with open(
        '/home/zt/pyProjects/JaneSt/Team/scripts/George/9-Weired-thoughts/9-3-lagging-days/models/lgbm_model_updated.pkl',
        'wb') as file:
    pickle.dump(model, file)
