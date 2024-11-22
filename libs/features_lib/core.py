"""

@authors: George, Raffaele
"""
from typing import Dict

import numpy as np
import pandas as pd

TRANSFORM_MAP = {'cyclic': np.diff,
                 'log': np.log,
                 'integrated': np.diff}

def transform_features(df: pd.DataFrame,
                       transf_dic: Dict[str, str]) -> (
        pd.DataFrame):
    df_transform = df.copy()
    feats = df_transform.columns.droplevel(1).unique()
    for c in feats:
        t = transf_dic[c]
        if transf_dic[c] in ['normal', 'fat']:
            pass
        elif transf_dic[c] == 'log':
            a = 1.01 * df_transform[c].min()
            df_tmp = df_transform[c] - a
            if df_tmp.min().min() < 0:
                raise ValueError('Something is off')
            df_transform[c] = TRANSFORM_MAP[t](df_tmp)
        elif transf_dic[c] in TRANSFORM_MAP.keys():
            df_transform[c] = TRANSFORM_MAP[t](df_transform[c])
        else:
            pass
    return df_transform
