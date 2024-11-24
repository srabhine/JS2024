import polars as pl, pandas as pd, numpy as np
import os
from gc import collect
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from sklearn.model_selection import train_test_split
import joblib


starting_date     = 500
target_name_str   = "responder_6"
random_state      = 50


data_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/Training_Responders/data"
important_features = ['date_id', 'time_id', 'symbol_id',
                      'feature_06', 'feature_07', 'feature_60', 'feature_36',
                       'feature_59', 'feature_08', 'feature_04', 'feature_05',
                       'feature_58', 'feature_38', 'feature_61', 'feature_01', 'feature_30',
                       'feature_75', 'feature_52', 'feature_56', 'feature_15', 'feature_33',
                       'feature_17', 'feature_76', 'feature_21', 'feature_31', 'feature_02',
                       'feature_23', 'feature_47', 'feature_50', 'feature_24', 'feature_78',
                       'feature_16', 'feature_22', 'feature_34', 'feature_77', 'feature_37',
                       'feature_68', 'feature_42', 'feature_45', 'feature_49', 'feature_48',
                       'feature_72', 'feature_20', 'feature_29',
                       'feature_69', 'feature_70', 'feature_25', 'feature_51', 'feature_55',
                       'feature_66']

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


def train_test_read(data_path):
    train_pl  = pl.scan_parquet(os.path.join(data_path, f"data_train.parquet"))
    sel_cols     = train_pl.collect_schema().names()
    sel_cols     = [c for c in sel_cols if c not in ["id", "partition_id"]]
    drop_cols    = [f"responder_{i}" for i in range(9)] + ["weight"]
    X_train       = train_pl.filter(pl.col("date_id").gt(starting_date)).select(pl.col(sel_cols)).collect(engine = "gpu").to_pandas()
    X_train.index = range(len(X_train))

    y_train                = X_train[target_name_str]
    sample_weights_train   = X_train["weight"].values.flatten()
    X_train                = X_train.drop(drop_cols , axis=1, errors = "ignore")




    test_pl      = pl.scan_parquet(os.path.join(data_path, f"data_test.parquet"))
    X_test       = test_pl.select(pl.col(sel_cols)).collect(engine = "gpu").to_pandas()
    X_test.index = range(len(X_test))
    y_test              = X_test[target_name_str]
    sample_weights_test = X_test["weight"].values.flatten()
    X_test              = X_test.drop(drop_cols , axis=1, errors = "ignore")
    return X_train, y_train, sample_weights_train, X_test, y_test, sample_weights_test


X_train, y_train, sample_weights_train, X_test, y_test, sample_weights_test = train_test_read(data_path)
X_train = X_train[important_features]
X_test = X_test[important_features]

# X_train, X_dev, y_train, y_dev, sample_weights_train, sample_weights_dev = (
#     train_test_split(X_train, y_train, sample_weights_train, test_size=0.2, random_state=random_state))



mymetric = CustomMetricMaker(method="LGB")


model = LGBMRegressor(
    device="gpu",
    objective="regression_l2",
    n_estimators=1500,
    max_depth=11,
    learning_rate=0.09908748900804956,
    colsample_bytree=0.6661445662478911,
    subsample=0.9761412327881754,
    random_state=random_state,
    reg_lambda=0.7931375224458609,
    reg_alpha=0.13595058518501274,
    num_leaves=884,
    verbosity=-1
)


model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
    eval_names=["Test"], eval_metric=[mymetric.make_metric], sample_weight=sample_weights_train,
    callbacks=[log_evaluation(100),early_stopping(100, verbose=True)])


y_train_pred = model.predict(X_train, num_iteration=model.best_iteration_)
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)

train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
dev_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

train_r2 = r2_score(y_train, y_train_pred, sample_weight=sample_weights_train)
dev_r2 = r2_score(y_test, y_test_pred, sample_weight=sample_weights_test)

print(f'Train LGB: {train_rmse:.4f}, R2: {train_r2:.4f}')
print(f'test  LGB: {dev_rmse:.4f}, R2: {dev_r2:.4f}')


version = "_4_pca"
# feature importance
joblib.dump(model, os.path.join("/home/zt/pyProjects/Optiver/JaneStreetMktPred/Training_Responders/trained_models", f"LGBM{version}.joblib"))
