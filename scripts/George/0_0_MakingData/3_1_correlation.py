import pandas as pd
import polars as pl
import numpy as np
import os
import random
import pickle
import matplotlib.pyplot as plt

is_linux = False
if is_linux:
    original_data_path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"

else:
    original_data_path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    feature_types_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"

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


feature_names = [f"feature_{i:02d}" for i in range(79)]
data = load_data(original_data_path, start_dt=1200, end_dt=1500)

corr_with_responder6 =  data[feature_names].corrwith(data['responder_6'])
high_correlation_features = corr_with_responder6[abs(corr_with_responder6) >= 0.05].index
high_cor_columns = corr_with_responder6.sort_values(ascending=False)[:10].index