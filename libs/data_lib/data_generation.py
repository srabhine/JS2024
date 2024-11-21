"""
George

"""

import pandas as pd
import polars as pl
from libs.io_lib.paths import DATA_DIR, TRAIN_DIR, \
    LAGS_FEATURES_TRAIN, LAGS_FEATURES_VALID
import numpy as np
import gc
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import StratifiedGroupKFold


class CONFIG:
    target_col = "responder_6"

    lag_cols_original = (["date_id", "symbol_id"] +
                         [f"responder_{idx}" for idx in range(9)])

    lag_cols_rename = { f"responder_{idx}":
                            f"responder_{idx}_lag_1"
                        for idx in range(9)}

    lag_feature_original = (["date_id", "time_id", "symbol_id"]
                            + [f"feature_{i:02d}" for i in range(79)])

    lag_feature_rename = { f"feature_{idx:02d}":
                               f"feature_{idx:02d}_lag_1"
                           for idx in range(79)}

    valid_ratio = 0.05
    start_dt = 1650


# Load training data
# Use last 2 parquets
train = pl.scan_parquet(TRAIN_DIR).select(
    pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
    pl.all(),).filter(
    pl.col("date_id").gt(CONFIG.start_dt))


# Create Lags data from training data
lags = train.select(pl.col(CONFIG.lag_cols_original))
lags = lags.rename(CONFIG.lag_cols_rename)
lags = lags.with_columns(
    date_id = pl.col('date_id') + 1,  # lagged by 1 day
    )
lags = lags.group_by(["date_id", "symbol_id"],
                     maintain_order=True).last()  # pick up last record of previous date


# Merge training data and lags data
train = train.join(lags, on=["date_id", "symbol_id"], how="left")

lag_feature = train.select(pl.col(CONFIG.lag_feature_original))
lag_feature = lag_feature.rename(CONFIG.lag_feature_rename)
lag_feature = lag_feature.with_columns(
    time_id = pl.col('time_id') + 1)
lag_feature = lag_feature.group_by(
    ["date_id", "time_id" ,"symbol_id"], maintain_order=True).last()  # pick up last record of previous date
train = train.join(lag_feature,
                   on=["date_id", "time_id" ,"symbol_id"],
                   how="left")


# Split training data and validation data
len_train   = train.select(pl.col("date_id")).collect().shape[0]
valid_records = int(len_train * CONFIG.valid_ratio)
len_ofl_mdl = len_train - valid_records
last_tr_dt  = train.select(pl.col("date_id")).collect().row(len_ofl_mdl)[0]

print(f"\n len_train = {len_train}")
print(f"\n len_ofl_mdl = {len_ofl_mdl}")
print(f"\n---> Last offline train date = {last_tr_dt}\n")

training_data = train.filter(pl.col("date_id").le(last_tr_dt))
validation_data = train.filter(pl.col("date_id").gt(last_tr_dt))

# Save data as parquets
training_data.collect().\
write_parquet(
    LAGS_FEATURES_TRAIN, partition_by ="date_id",
)

validation_data.collect().\
write_parquet(
    LAGS_FEATURES_VALID, partition_by ="date_id",
)
