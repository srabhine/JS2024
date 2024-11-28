import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import polars as pl
from typing import Optional, List, Union, Dict, Any
import os
import pickle
import numpy as np


def get_features_classification(file_path: Optional[Any] = None):
    if file_path is None:
        raise ValueError
    feat_types = pd.read_csv(file_path, index_col=0)
    return feat_types.to_dict()['Type']


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
    start_dt = 500
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

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0)
    return data


def get_norm_features_dict():
    # file_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"
    file_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"
    feat_types_dic = get_features_classification(file_path)
    features_to_scale = [feature for feature, ftype in feat_types_dic.items() if ftype == 'normal']
    return features_to_scale


def save_scalers(features_to_scale):
    # scaler_dir = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save"
    scaler_dir = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers"
    os.makedirs(scaler_dir, exist_ok=True)
    # Dictionary to hold all scalers
    scalers = {}

    # Group by 'symbol_id' and apply scaling
    for symbol_id, group in data.groupby('symbol_id'):
        # Initialize MinMaxScaler for this group
        scaler = StandardScaler()

        # Fit the scaler on the group's "normal" features
        scaler.fit(group[features_to_scale])

        # Store the scaler in the dictionary with symbol_id as the key
        scalers[symbol_id] = scaler

    # Save all scalers to a single file
    scaler_filename = f'{scaler_dir}/all_scalers.pkl'

    with open(scaler_filename, 'wb') as f:
        pickle.dump(scalers, f)

data = load_data()
features_to_scale = get_norm_features_dict()
save_scalers(features_to_scale)

"""
with open(scaler_filename, 'rb') as f:
    scaler = pickle.load(f)

df_validation[features_to_scale] = scaler.transform(df_validation[features_to_scale])
"""

"""
features_to_scale =
['feature_01',
 'feature_04',
 'feature_18',
 'feature_19',
 'feature_33',
 'feature_36',
 'feature_39',
 'feature_40',
 'feature_41',
 'feature_42',
 'feature_43',
 'feature_44',
 'feature_45',
 'feature_46',
 'feature_50',
 'feature_51',
 'feature_52',
 'feature_53',
 'feature_54',
 'feature_55',
 'feature_56',
 'feature_57',
 'feature_63',
 'feature_64',
 'feature_78']
"""
