import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import polars as pl
from typing import Optional, List, Union, Dict, Any, Tuple
import os
import pickle
import numpy as np



SYMBOLS = list(range(39))
RESPONDERS = list(range(9))
IX_IDS_BY_SYM = ['symbol_id', 'date_id', 'time_id']
FEATS = [f"feature_{i:02d}" for i in range(79)]
TARGET = 'responder_6'


#
# path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet/partition_id=5"
# data = pl.scan_parquet(path).collect().to_pandas()

def load_data():
    path_win = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    path_linux = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    start_dt = 600
    end_dt = 1400
    data = pl.scan_parquet(path_linux
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




features_to_scale = ['feature_01', 'feature_04','feature_18','feature_19','feature_33','feature_36','feature_39','feature_40',
                     'feature_41','feature_42','feature_43', 'feature_44','feature_45','feature_46','feature_50','feature_51',
                     'feature_52','feature_53','feature_54','feature_55','feature_56','feature_57','feature_63','feature_64',
                     'feature_78']


def normalize_and_save_scalers(data: pd.DataFrame, features_to_scale: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Calculate mean and std for each feature to be scaled
    means = data[features_to_scale].mean()
    stds = data[features_to_scale].std()

    # Scale the specified features directly in the original DataFrame
    data[features_to_scale] = (data[features_to_scale] - means) / stds

    # Create a DataFrame for scalers
    scalers_df = pd.DataFrame({'mean': means, 'std': stds})

    # Return the modified DataFrame and the scalers DataFrame
    return data, scalers_df

data = load_data()
data = data.fillna(0)
data, scalers_df = normalize_and_save_scalers(data, features_to_scale)
pickle_file = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_whole.pkl"
with open(pickle_file, 'wb') as f:
    pickle.dump(scalers_df, f)





