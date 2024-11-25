"""

@author: Raffaele M Ghigliazza
"""
from pathlib import Path
from typing import Union, Any, Optional

import pandas as pd
import polars as pl

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

