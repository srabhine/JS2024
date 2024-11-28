"""

@author: Raffaele M Ghigliazza
"""
from typing import List, Dict, Optional, Union, Any

import pandas as pd
import polars as pl

from data_lib.variables import TARGET, SYMBOLS
from features_lib.core import transform_features
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


def get_features_classification(file_path: Optional[Any] = None):
    if file_path is None:
        file_path = DATA_DIR / 'features_types.csv'
    feat_types = pd.read_csv(file_path, index_col=0)
    feat_types_dic_tmp = feat_types.to_dict()['Type']
    feat_types_dic = {f'feature_{k:02d}': v for k, v in
                      feat_types_dic_tmp.items()}
    return feat_types_dic


def get_data_by_symbol(feature_names: List,
                       sym: Optional[Union[int, List]] = None,
                       is_transform: bool = False,
                       feat_types_dic: Optional[Dict] = None,):
    # Load data
    df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
    vld = pl.scan_parquet(LAGS_FEATURES_VALID).collect().to_pandas()

    if is_transform:
        print('Transforming data')
        df, scalers_mu, scalers_sg = transform_features(
            df, feat_types_dic)
        vld, *_ = transform_features(df, feat_types_dic)
        scalers_mu.to_csv(DATA_DIR / 'scalers_mu.csv')
        scalers_sg.to_csv(DATA_DIR / 'scalers_sg.csv')

    # Select subset
    if sym is None:
        sym = SYMBOLS
        df_sym = df.query(f'symbol_id in {sym}')
        vld_sym = vld.query(f'symbol_id in {sym}')
    elif isinstance(sym, int):
        df_sym = df[df['symbol_id'] == sym]
        vld_sym = vld[vld['symbol_id'] == sym]
    elif len(sym) == len(SYMBOLS):
        df_sym = df
        vld_sym = vld
    else:
        df_sym = df.query(f'symbol_id in {sym}')
        vld_sym = vld.query(f'symbol_id in {sym}')
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

