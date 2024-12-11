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


# Categorical Features
categorical_features = ['feature_09','feature_10','feature_11']

data = load_data(original_data_path, start_dt=1450, end_dt=1500)
features = ['feature_12', 'feature_13', 'feature_14', 'feature_16', 'feature_17',
           'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_71',
           'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76',
           'feature_77']
def transf_moving_avg(data, days=10, features=['feature_12']):
    for name, sym_df in data.groupby('symbol_id'):
        data.loc[sym_df.index, features] = sym_df[features].rolling(window=days, min_periods=1).mean().astype('float32')
    