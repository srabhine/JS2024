import pandas as pd
import polars as pl
import numpy as np
import gc
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import StratifiedGroupKFold

class CONFIG:
    target_col = "responder_6"
    lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{idx}" for idx in range(9)]
    lag_cols_rename = { f"responder_{idx}" : f"responder_{idx}_lag_1" for idx in range(9)}
    valid_ratio = 0.05
    start_dt = 500


train = pl.scan_parquet(
    f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
).select(
    pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
    pl.all(),
).with_columns(
    (pl.col(CONFIG.target_col)*2).cast(pl.Int32).alias("label"), # multiply target by 2 and store it as label
).filter(
    pl.col("date_id").gt(CONFIG.start_dt)
)

# Create Lags data from training data
lags = train.select(pl.col(CONFIG.lag_cols_original))
lags = lags.rename(CONFIG.lag_cols_rename)
lags = lags.with_columns(
    date_id = pl.col('date_id') + 1,  # lagged by 1 day
    )
lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last()  # pick up last record of previous date


# Merge training data and lags data
train = train.join(lags, on=["date_id", "symbol_id"],  how="left")


# Split training data and validation data
len_train   = train.select(pl.col("date_id")).collect().shape[0]
valid_records = int(len_train * CONFIG.valid_ratio)
len_ofl_mdl = len_train - valid_records
last_tr_dt  = train.select(pl.col("date_id")).collect().row(len_ofl_mdl)[0]

print(f"\n len_train = {len_train}")
print(f"\n len_ofl_mdl = {len_ofl_mdl}")
print(f"\n---> Last offline train date = {last_tr_dt}\n")

training_data = train.filter(pl.col("date_id").le(last_tr_dt))
validation_data   = train.filter(pl.col("date_id").gt(last_tr_dt))


# Save data as parquets
training_data.collect().\
write_parquet(
    f"/home/zt/pyProjects/JaneSt/Team/data/CustomData_1_RespLags/training", partition_by = "date_id",
)

validation_data.collect().\
write_parquet(
    "/home/zt/pyProjects/JaneSt/Team/data/CustomData_1_RespLags/validation", partition_by = "date_id",
)