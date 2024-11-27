"""

For submission: copy and paste the whole file except the '__main__'

@author: Raffaele M Ghigliazza
"""
import copy
from typing import Optional, List, Union, Dict

import numpy as np
import pandas as pd

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
    data_by_sym.ffill().fillna(0)

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


def transform_data(
        data_all: pd.DataFrame,
        feature_names: Optional[List] = None,
        transformation: Optional[Union[str,
        Dict[str, str]]] = None,
        params: Optional[Dict] = None):
    data_by_sym = stack_features_by_sym(data_all,
                                        feature_names=feature_names)
    # (date_id, time_id) x (features+, 'symbol_id')
    transf_dic = get_transformation_dic(data_all,
                                        transformation=transformation)
    feat_names = data_by_sym.columns.droplevel(1).unique()
    symbol_ids = data_by_sym.columns.droplevel(0).unique()
    no_params = params is None
    if no_params:
        params = {}
    data_transformed = data_by_sym.copy()
    for f in feat_names:
        if no_params:
            params[f] = {}
        if f in transf_dic:
            if transf_dic[f] in ['norm', 'normal']:
                if no_params:
                    params[f] = {'loc': data_by_sym[f].mean(),
                                 'scale': data_by_sym[f].std()}
                for s in symbol_ids:
                    data_transformed[(f, s)] = \
                        ((data_by_sym[(f, s)] - params[f]['loc'][s])
                         / params[f]['scale'][s])
            elif transf_dic[f] == 'log':
                if no_params:
                    params[f] = {'shift': pd.Series(index=symbol_ids)}
                for s in symbol_ids:
                    ds = data_by_sym[(f, s)]
                    a = 1.1 * ds.min() if ds.min() < 0 else 0.9
                    params[f]['shift'].loc[s] = a
                    data_transformed[(f, s)] = \
                        np.log(data_by_sym[(f, s)] - a)
            elif transf_dic[f] == '0_1':
                if no_params:
                    params[f] = {'min': data_by_sym[f].min(),
                                 'max': data_by_sym[f].max()}
                for s in symbol_ids:
                    ds = data_by_sym[(f, s)]
                    mn = params[f]['min'].loc[s]
                    mx = params[f]['max'].loc[s]
                    data_transformed[(f, s)] = (ds - mn) / (mx - mn)
            elif transf_dic[f] == '-1_1':
                if no_params:
                    params[f] = {'min': data_by_sym[f].min(),
                                 'max': data_by_sym[f].max()}
                for s in symbol_ids:
                    ds = data_by_sym[(f, s)]
                    mn = params[f]['min'].loc[s]
                    mx = params[f]['max'].loc[s]
                    data_transformed[(f, s)] = \
                        2 * (ds - mn) / (mx - mn) - 1
            elif transf_dic[f] == 'tanh':
                if no_params:
                    params[f] = {}
                for s in symbol_ids:
                    data_transformed[(f, s)] = \
                        np.tanh(data_by_sym[(f, s)])
    data_transformed = unstack_features_by_sym(
        data_transformed)
    return data_transformed, params

# # to only normalize selected data
# # params dict is hard to use, so not using it for now.
# def transform_data_2(
#         data_all: pd.DataFrame,
#         feature_names: Optional[List] = None,
#         transformation: Optional[Union[str,
#         Dict[str, str]]] = None):
#     data_by_sym = stack_features_by_sym(data_all,feature_names=feature_names)
#     # (date_id, time_id) x (features+, 'symbol_id')
#     transf_dic = get_transformation_dic(data_all,transformation=transformation)
#     data_transformed = data_by_sym.copy()
#     scalers_mu = data_transformed.mean(axis=0)
#     scalers_sg = data_transformed.std(axis=0)
#     cols = data_transformed.columns.droplevel(1).unique()
#     for c in cols:
#         if transf_dic[c] in ['norm', 'normal']:
#             if c not in ['symbol_id', 'date_id','time_id', 'weight', 'responder_6']:
#                 data_transformed[c] = (data_transformed[c] - data_transformed[c].mean()) / data_transformed[c].std()
#
#     mix = pd.MultiIndex.from_tuples([c[1], c[0]] for c in data_transformed.columns)
#     data_transformed.columns = mix
#     data_transformed = data_transformed.stack(level=0, future_stack=True)
#     data_transformed.index.names = ['date_id', 'time_id', 'symbol_id']
#     data_transformed.reset_index(inplace=True)
#     # df_transform['weight'] = df_transform['weight'].fillna(0)  #
#     # # still not enough ...
#     data_transformed.fillna(0, inplace=True)
#     if len(data_transformed) == len(data_all):
#         data_transformed.index = data_all.index
#     else:
#         print('Warning: transformation added rows')
#
#
#     return data_transformed, scalers_mu, scalers_sg

if __name__ == '__main__':
    import polars as pl
    from matplotlib import pyplot as plt
    from data_lib.datasets import get_features_classification
    from io_lib.paths import LAGS_FEATURES_TRAIN

    data_tmp = pl.scan_parquet(
        LAGS_FEATURES_TRAIN).collect().to_pandas()

    # Cleanup
    data_by_sym = stack_features_by_sym(data_tmp)
    data_cleaned = unstack_features_by_sym(data_by_sym)
    print(data_tmp.shape, data_cleaned.shape)

    # Normalization
    s_mu, s_sg = calc_scalers(data_tmp, transformation='norm')
    s_mu, s_sg = calc_scalers(data_tmp,
                              transformation={'feature_71': 'norm',
                                              'responder_6': 'none'},
                                  )

    # Transformation
    feat_types_dic = get_features_classification()
    feat_types_dic['responder_6'] = 'none'
    feat_types_dic['responder_6'] = '0_1'
    feat_types_dic['responder_6'] = '-1_1'
    feat_types_dic['responder_6'] = 'tanh'
    data_transf, params = transform_data(
        data_tmp, transformation=feat_types_dic)

    print(params)

    # data_org = data_tmp.set_index(['date_id', 'time_id',
    #                                        'symbol_id'])
    # data_t = data_transf.set_index(['date_id', 'time_id',
    #                                        'symbol_id'])
    # data_plot = pd.concat((data_org[TARGET], data_t[TARGET]),
    #                       axis=1, keys=['raw', 'transf']).dropna()
    #
    # f, ax = plt.subplots()
    # ax.scatter(data_plot.iloc[:, 0], data_plot.iloc[:, 1])
    # plt.show()

    # During test submission or validation
    # Assume a given (date_id, time_id, symbol_id, weight) and make
    # sure they are in the columns
    # Prepare the data in rectangular form
    cols = IX_IDS_BY_SYM + ['weight'] + FEATS
    df_test = pd.DataFrame(np.random.randn(len(SYMBOLS), len(cols)),
                           columns=cols)
    df_test['date_id'] = 1678
    df_test['time_id'] = 23
    df_test['symbol_id'] = SYMBOLS
    df_test_transf, *_ = transform_data(
        df_test, transformation=feat_types_dic,
        params=params)
    print(df_test_transf)
