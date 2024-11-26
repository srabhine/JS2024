import pandas as pd
import polars as pl
import numpy as np

start_date = 0
last_train_date = 1600
feature_names = (["symbol_id", "date_id", "time_id", "weight"] +
                 [f"feature_{i:02d}" for i in range(79)] +
                 [f"responder_6"])


def get_all_data(start_date):
    data = pl.scan_parquet(
        f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    ).select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
        pl.all(),
    ).filter(
        pl.col("date_id").gt(start_date)
    )
    return data # lazy dataframe


def get_last_day_data(data):
    data = data.select(pl.col(feature_names))
    data = data.group_by(["symbol_id", "date_id"], maintain_order=True).last()  # pick up last record of previous date
    return data


def get_mean_data(data):
    data = data.select(pl.col(feature_names))
    agg_columns = [pl.mean(col).alias(f"mean_{col}") for col in feature_names if col not in ["symbol_id", "date_id", "time_id"]]
    data = data.group_by(["symbol_id", "date_id"]).agg(agg_columns)
    return data



data = get_all_data(start_date)

data = get_mean_data(data)


data.collect().write_parquet(
    f"/home/zt/pyProjects/JaneSt/Team/data/mean_data/train.parquet"
)

