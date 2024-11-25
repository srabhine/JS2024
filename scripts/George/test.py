import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import Dict, Optional, Union

from pathlib import Path

IX_IDS_BY_SYM = ['symbol_id', 'date_id', 'time_id']
FEATS = [f"feature_{i:02d}" for i in range(79)]
TARGET = 'responder_6'


DATA_DIR = "E:\Python_Projects\JS2024\GITHUB_C\data\\"

LAGS_FEATURES_TRAIN = "E:\Python_Projects\JS2024\GITHUB_C\data\lags_features\\training_parquet"
LAGS_FEATURES_VALID = "E:\Python_Projects\JS2024\GITHUB_C\data\lags_features\\validation_parquet"
def get_features_classification():
    feat_types = pd.read_csv(DATA_DIR + 'features_types.csv',
                             index_col=0)
    feat_types_dic_tmp = feat_types.to_dict()['Type']
    feat_types_dic = {f'feature_{k:02d}': v for k, v in
                      feat_types_dic_tmp.items()}
    return feat_types_dic





feat_types_dic = get_features_classification()
feature_names = FEATS

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
vld = pl.scan_parquet(LAGS_FEATURES_VALID).collect().to_pandas()

data = df


data = data[IX_IDS_BY_SYM + ['weight'] + FEATS + [TARGET]]
data.set_index(IX_IDS_BY_SYM, append=True, drop=True, inplace=True)
"""
data.shape
Out[3]: (2073456, 81)
"""
data = data.droplevel(0, axis='index') # dropped the original index
"""
data.shape
Out[5]: (2073456, 81)
"""
data = data.unstack(level=['symbol_id']) # unstacked the symbol_id
"""
(54208, 3159)
3159 % 81 =0
each symbol id has its own 81 columns
"""

cols = data.columns.droplevel(1).unique()
feat_types_dic = get_features_classification()

TRANSFORM_MAP = {'cyclic': lambda x: x,
                 'log': np.log,
                 'integrated': lambda x: x,}
for c in cols:
    if feat_types_dic is None:
        if c not in ['weight']:
            data[c] = (data[c] - data[
                c].mean()) / data[c].std()
    elif feat_types_dic == 'cleanup':
        if feat_types_dic in FEATS + [TARGET]:
            pass
    else:
        if c in list(feat_types_dic.keys()):
            t = feat_types_dic[c]
            if feat_types_dic[c] in ['normal', 'fat']:
                pass
            elif feat_types_dic[c] == 'log':
                x_min = data[c].min()
                if x_min.max() < 0:
                    a = 1.01 * x_min
                    df_tmp = data[c] - a
                elif x_min.min() > 0:
                    a = 0.99 * x_min
                    df_tmp = data[c] - a
                else:
                    df_tmp = data[c]
                if df_tmp.min().min() < 0:
                    raise ValueError('Something is off')
                data[c] = TRANSFORM_MAP[t](df_tmp)
            elif feat_types_dic[c] in TRANSFORM_MAP.keys():
                data[c] = (
                    TRANSFORM_MAP[t](data[c]))
            else:
                pass

mix = pd.MultiIndex.from_tuples([c[1], c[0]] for c in data.columns)
data.columns = mix
data = data.stack(level=0, future_stack=True)
data.index.names = ['date_id', 'time_id', 'symbol_id']
data.reset_index(inplace=True)
# df_transform['weight'] = df_transform['weight'].fillna(0)  #
# # still not enough ...
data.fillna(0, inplace=True)
