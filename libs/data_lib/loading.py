"""

@author: Raffaele M Ghigliazza
"""
from pathlib import Path
from typing import Union, Any, Optional

import pandas as pd
import polars as pl

from data_lib.variables import FEATS, TARGET
from io_lib.paths import DATA_DIR


def load_scalers_core(fnm: str,
                      data_dir: Optional[Union[str, Any]] = None):
    if data_dir is None:
        data_dir = DATA_DIR
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / fnm, header=1,
                     names=['feature', 'symbol_id', 'value'])
    transformed_df = df.pivot(index='symbol_id', columns='feature',
                              values='value')
    return transformed_df


def load_scalers(data_dir: Optional[Union[str, Any]] = None,
                 fillna: bool = False):
    scalers_mu = load_scalers_core('scalers_mu.csv',
                                   data_dir=data_dir)
    scalers_mu = scalers_mu.drop('weight', axis=1)
    scalers_sg = load_scalers_core('scalers_sg.csv',
                                   data_dir=data_dir)
    scalers_sg = scalers_sg.drop('weight', axis=1)
    if fillna:
        scalers_mu.fillna(0.0, inplace=True)
        scalers_sg.fillna(1.0, inplace=True)
    return scalers_mu, scalers_sg


def normalize_test_data(test_data: pd.DataFrame,
                        scalers_mu: pd.DataFrame,
                        scalers_sg: pd.DataFrame,
                        fillna: bool = True):
    """
    Notes:
        scalers_mu and scalers_sg are returned and could be
        different from the input versions. For example, if a new
        symbol appears in test_data, they must be augmented

    Args:
        test_data:
        scalers_mu:
        scalers_sg:
        fillna:

    Returns:

    """
    # test_data.index = list(test_data.index)
    test_tmp = test_data.copy()
    test_tmp.drop(['row_id', 'date_id', 'time_id',
                   'weight', 'is_scored'], axis=1,
                  inplace=True)
    test_tmp.set_index(['symbol_id'], drop=True, inplace=True)
    index_org = test_data.index
    data_tmp = pd.concat([test_tmp, scalers_mu, scalers_sg],
                         axis=1, keys=['test', 'mu', 'sg'])
    data_tmp = data_tmp.loc[index_org]
    if fillna:
        test_tmp = data_tmp['test'].fillna(0.0)  # careful! No responder here
        scalers_mu = data_tmp['mu'].fillna(0.0)
        scalers_sg = data_tmp['sg'].fillna(1.0)
    else:
        test_tmp = data_tmp['test']  # careful! No responder here
        scalers_mu = data_tmp['mu']
        scalers_sg = data_tmp['sg']


    test_norm = ((test_tmp[FEATS] - scalers_mu[FEATS]) /
                 scalers_sg[FEATS])

    return test_norm, scalers_mu, scalers_sg


def denormalize_predictions(pred_norm: pd.Series,
                            scalers_mu: pd.DataFrame,
                            scalers_sg: pd.DataFrame):
    pred_norm.name = TARGET
    return (pred_norm + scalers_mu[TARGET]) * scalers_sg[TARGET]


