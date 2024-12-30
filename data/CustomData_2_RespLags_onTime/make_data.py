import pandas as pd
import polars as pl
import numpy as np
import gc
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import StratifiedGroupKFold

class CONFIG:
    target_col = "responder_6"
    lag_cols_original = ["symbol_id", "date_id", "time_id"] + [f"responder_{idx}" for idx in range(9)]
    lag_cols_rename = { f"responder_{idx}" : f"responder_{idx}_lag_1" for idx in range(9)}
    valid_ratio = 0.05
    start_dt = 500


train = pl.scan_parquet(
    # f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    "E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
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
lags = lags.group_by(["symbol_id","date_id", "time_id"], maintain_order=True).last()  # pick up last record of previous date



# Merge training data and lags data
train = train.join(lags, on=["symbol_id","date_id", "time_id"],  how="left")




# Save data as parquets
train.collect().write_parquet(
    f"E:\Python_Projects\JS2024\GITHUB_C\data\CustomData_2_RespLags_onTime\\trainData", partition_by = "date_id",
)
