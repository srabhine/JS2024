"""

@authors: George, Raffaele
"""
from typing import Dict

import numpy as np
import pandas as pd

from data_lib.variables import IX_IDS_BY_SYM, FEATS

TRANSFORM_MAP = {'cyclic': np.diff,
                 'log': np.log,
                 'integrated': np.diff}


def transform_features(data_all: pd.DataFrame,
                       transf_dic: Dict[str, str]) -> (
        pd.DataFrame):
    data = data_all[IX_IDS_BY_SYM + FEATS].copy()
    data.set_index(IX_IDS_BY_SYM, append=True,
                   drop=True, inplace=True)
    data = data.droplevel(0, axis='index')
    data = data.unstack(level=['symbol_id'])

    df_transform = data.copy()
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

    mix = pd.MultiIndex.from_tuples(
        [c[1], c[0]] for c in df_transform.columns)
    df_transform.columns = mix
    df_transform = df_transform.stack(level=0)

    return df_transform
