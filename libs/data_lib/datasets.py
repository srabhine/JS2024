"""

@author: Raffaele M Ghigliazza
"""
from typing import List

import pandas as pd
import polars as pl

from data_lib.variables import TARGET
from io_lib.paths import DATA_DIR, LAGS_FEATURES_TRAIN, \
    LAGS_FEATURES_VALID


def get_indexed_dataset():
    data = pd.read_parquet(DATA_DIR / 'train_parquet' /
                           'partition_id=9' /
                           'part-0.parquet',
                           engine='pyarrow')

    # Group
    data_ixd = data.copy()
    data_ixd.set_index(['symbol_id', 'date_id', 'time_id'],
                       inplace=True)
    data_ixd.sort_index(axis=0, inplace=True)

    return data_ixd


def get_symbols_dataset(sym: int = 1):
    data_ixd = get_indexed_dataset()
    data_sym = data_ixd.loc[sym]
    return data_sym


def get_data_by_symbol(feature_names: List,
                       sym: int = 1):
    # Load data
    df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
    vld = pl.scan_parquet(LAGS_FEATURES_VALID).collect().to_pandas()

    # Select subset
    df_sym = df[df['symbol_id'] == sym].copy()
    vld_sym = vld[vld['symbol_id'] == sym].copy()
    print(df_sym.head())

    df_sym[feature_names] = df_sym[feature_names].ffill().fillna(0)
    vld_sym[feature_names] = vld_sym[feature_names].ffill().fillna(0)

    # Prepare datasets
    X_train = df_sym[feature_names]
    y_train = df_sym[TARGET]
    w_train = df_sym['weight']
    X_valid = vld_sym[feature_names]
    y_valid = vld_sym[TARGET]
    w_valid = vld_sym['weight']

    return (df_sym, vld_sym, X_train, y_train, w_train, X_valid,
            y_valid, w_valid)

