"""

@authors: George, Raffaele
"""
from typing import Dict, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from data_lib.variables import IX_IDS_BY_SYM, FEATS, TARGET

# TRANSFORM_MAP = {'cyclic': np.diff,
#                  'log': np.log,
#                  'integrated': np.diff}
#

# TRANSFORM_MAP = {'cyclic': lambda x: x,
#                  'log': lambda x: x,
#                  'integrated': np.diff}

# TRANSFORM_MAP = {'cyclic': np.diff,
#                  'log': lambda x: x,
#                  'integrated': lambda x: x,}

TRANSFORM_MAP = {'cyclic': lambda x: x,
                 'log': np.log,
                 'integrated': lambda x: x,}

def transform_features(data_all: pd.DataFrame,
                       transf_dic: Optional[Union[str, Dict[str,
                       str]]] = None) -> tuple[Any, Any, Any]:
    data = data_all[IX_IDS_BY_SYM + ['weight'] + FEATS + [
        TARGET]]
    data.set_index(IX_IDS_BY_SYM, append=True,
                   drop=True, inplace=True)
    data = data.droplevel(0, axis='index')
    data = data.unstack(level=['symbol_id'])

    df_transform = data.copy()
    cols = df_transform.columns.droplevel(1).unique()
    scalers_mu = df_transform.mean(axis=0)
    scalers_sg = df_transform.std(axis=0)
    # Careful because 'weight' is also in the columns
    # and that is not meaningful
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
    # df_transform['weight'] = df_transform['weight'].fillna(0)  #
    # # still not enough ...
    df_transform.fillna(0, inplace=True)
    if len(df_transform) == len(data_all):
        df_transform.index = data_all.index
    else:
        print('Warning: transformation added rows')

    return df_transform, scalers_mu, scalers_sg
