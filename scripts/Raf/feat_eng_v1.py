"""

@authors: George, Raffaele
"""
from typing import Dict

import numpy as np
import pandas as pd
import polars as pl

from data_lib.variables import TARGET, FEATS, IX_IDS_BY_SYM
from features_lib.core import transform_features
from io_lib.paths import DATA_DIR, LAGS_FEATURES_TRAIN

feat_types = pd.read_csv(DATA_DIR / 'features_types.csv',
                         index_col=0)
feat_types_dic_tmp = feat_types.to_dict()['Type']
feat_types_dic = {f'feature_{k:02d}':v for k, v in
                  feat_types_dic_tmp.items()}

data_all = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
data = data_all[IX_IDS_BY_SYM + FEATS].copy()
data.set_index(IX_IDS_BY_SYM, append=True,
               drop=True, inplace=True)
data = data.droplevel(0, axis='index')
data = data.unstack(level=['symbol_id'])


data_transform = transform_features(data, feat_types_dic)
mix = pd.MultiIndex.from_tuples([c[1], c[0]] for c in data_transform.columns)
data_transform.columns = mix
data_tmp = data_transform.stack(level=0)
print(data_tmp)

