"""

@author: Raffaele M Ghigliazza
"""
import copy
from typing import Optional, List, Union, Dict

import numpy as np
import pandas as pd
import polars as pl


SYMBOLS = list(range(39))
RESPONDERS = list(range(9))
IX_IDS_BY_SYM = ['symbol_id', 'date_id', 'time_id']
FEATS = [f"feature_{i:02d}" for i in range(79)]
TARGET = 'responder_6'


def merge_dic(a, b):
    """
    Merge dictionary 'b' into dictionary 'a'. This is a wrapper on
    merge_dic_core so that it is not changed in place

    Notes
    -----
    In case b has the same key as a, the value will be overwritten

    Parameters
    ----------
    a: Dict
    b: Dict

    Returns
    -------
    c: Dict
        Merged dictionary
    """
    return _merge_dic_core(copy.deepcopy(a), b)


def _merge_dic_core(a, b):
    """
    Merge dictionary b into dictionary a

    Notes
    -----
    1) The dictionary a will be changed in place
    2) See: https://stackoverflow.com/questions/7204805/how-to-merge-
    dictionaries-of-dictionaries/7205107#7205107

    Parameters
    ----------
    a: Dict
    b: Dict

    Returns
    -------

    """
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge_dic_core(a[key], b[key])
            else:
                a[key] = b[key]  # update
        else:
            a[key] = b[key]
    return a


def check_cols(cols: List, df: pd.DataFrame) -> List:
    cols_valid = []
    for col in cols:
        if col in df.columns:
            cols_valid.append(col)
    return cols_valid


def stack_features_by_sym(data_all: pd.DataFrame,
                          feature_names: Optional[List] = None):
    if feature_names is None:
        feature_names = FEATS
    cols = IX_IDS_BY_SYM + ['weight'] + feature_names + [TARGET]
    cols = check_cols(cols, data_all)
    data_by_sym = data_all[cols]
    data_by_sym.set_index(IX_IDS_BY_SYM, append=True, drop=True,
                   inplace=True)
    data_by_sym = data_by_sym.droplevel(0, axis='index')
    data_by_sym = data_by_sym.unstack(level=['symbol_id'])

    return data_by_sym


def unstack_features_by_sym(data_by_sym: pd.DataFrame):
    mix = pd.MultiIndex.from_tuples(
        [c[1], c[0]] for c in data_by_sym.columns)
    data_by_sym.columns = mix
    data_by_sym = data_by_sym.stack(level=0, future_stack=True)
    data_by_sym.index.names = ['date_id', 'time_id', 'symbol_id']
    data_by_sym.reset_index(inplace=True)
    return data_by_sym


def get_transformation_dic(data_all: pd.DataFrame,
                           transformation: Optional[Union[str,
                           Dict[str, str]]] = None):
    transf_dic_base = {c: 'none' for c in data_all.columns}
    if isinstance(transformation, str):
        transf_dic = {c: transformation for c in data_all.columns}
    elif isinstance(transformation, Dict):
        transf_dic = merge_dic(transf_dic_base, transformation)
    else:
        transf_dic = transf_dic_base
    return transf_dic


def calc_scalers(
        data_all: pd.DataFrame,
        feature_names: Optional[List] = None,
        transformation: Optional[Union[str,
        Dict[str, str]]] = None):
    data_by_sym = stack_features_by_sym(data_all,
                                        feature_names=feature_names)
    # (date_id, time_id) x (features+, 'symbol_id')
    transf_dic = get_transformation_dic(data_all,
                                        transformation=transformation)
    feat_names = data_by_sym.columns.droplevel(1).unique()
    symbol_ids = data_by_sym.columns.droplevel(0).unique()
    scalers_mu = pd.DataFrame(columns=feat_names, index=symbol_ids)
    scalers_sg = pd.DataFrame(columns=feat_names, index=symbol_ids)
    for f in feat_names:
        scalers_mu[f] = 0
        scalers_sg[f] = 1
        if f in transf_dic and transf_dic[f] == 'norm':
            scalers_mu[f] = data_by_sym[f].mean()
            scalers_sg[f] = data_by_sym[f].std()
    return scalers_mu, scalers_sg


if __name__ == '__main__':
    from data_lib.datasets import get_features_classification
    from io_lib.paths import LAGS_FEATURES_TRAIN

    feat_types_dic = get_features_classification()
    data_tmp = pl.scan_parquet(
        LAGS_FEATURES_TRAIN).collect().to_pandas()
    s_mu, s_sg = calc_scalers(data_tmp, transformation='norm')
    s_mu, s_sg = calc_scalers(data_tmp,
                              transformation={'feature_71': 'norm',
                                              'responder_6': 'none'},
                              )
