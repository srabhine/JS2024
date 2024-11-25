import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score







IX_IDS_BY_SYM = ['symbol_id', 'date_id', 'time_id']
FEATS = [f"feature_{i:02d}" for i in range(79)]
TARGET = 'responder_6'





def get_features_classification():
    feat_types = pd.read_csv(DATA_DIR / 'features_types.csv',
                             index_col=0)
    feat_types_dic_tmp = feat_types.to_dict()['Type']
    feat_types_dic = {f'feature_{k:02d}': v for k, v in
                      feat_types_dic_tmp.items()}
    return feat_types_dic




data = data[IX_IDS_BY_SYM + ['weight'] + FEATS + [TARGET]]
data.set_index(IX_IDS_BY_SYM, append=True,drop=True, inplace=True)
data = data.droplevel(0, axis='index')
data = data.unstack(level=['symbol_id'])

df_transform = data.copy()
cols = df_transform.columns.droplevel(1).unique()

for c in cols:
    if transf_dic is None:
        if c not in ['weight']:
            df_transform[c] = (df_transform[c] - df_transform[
                c].mean()) / df_transform[c].std()
    elif transf_dic == 'cleanup':
        if transf_dic in FEATS + [TARGET]:
            pass
    else:
        if c in list(transf_dic.keys()):
            t = transf_dic[c]
            if transf_dic[c] in ['normal', 'fat']:
                pass
            elif transf_dic[c] == 'log':
                x_min = df_transform[c].min()
                if x_min.max() < 0:
                    a = 1.01 * x_min
                    df_tmp = df_transform[c] - a
                elif x_min.min() > 0:
                    a = 0.99 * x_min
                    df_tmp = df_transform[c] - a
                else:
                    df_tmp = df_transform[c]
                if df_tmp.min().min() < 0:
                    raise ValueError('Something is off')
                df_transform[c] = TRANSFORM_MAP[t](df_tmp)
            elif transf_dic[c] in TRANSFORM_MAP.keys():
                df_transform[c] = (
                    TRANSFORM_MAP[t](df_transform[c]))
            else:
                pass

mix = pd.MultiIndex.from_tuples(
    [c[1], c[0]] for c in df_transform.columns)
df_transform.columns = mix
df_transform = df_transform.stack(level=0, future_stack=True)
df_transform.index.names = ['date_id', 'time_id', 'symbol_id']
df_transform.reset_index(inplace=True)
if len(df_transform) == len(data_all):
    df_transform.index = data_all.index
else:
    print('Warning: transformation added rows')