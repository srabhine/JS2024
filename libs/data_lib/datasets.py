"""

@author: Raffaele M Ghigliazza
"""
from typing import List

import pandas as pd

from io_lib.paths import DATA_DIR, LAGS_FEATURES_TRAINING


def get_indexed_dataset():
    data = pd.read_parquet(DATA_DIR / 'train_parquet' /
                           'partition_id=9' /
                           'part-0.parquet',
                           engine='pyarrow')

    # Group
    data_ixd = data.copy()
    data_ixd.set_index(['symbol_id', 'date_id', 'time_id'],
                       inplace=True)
    data_ixd.sort_index(axis=0, inplace=True)

    return data_ixd


def get_symbols_dataset(sym: int = 1):
    data_ixd = get_indexed_dataset()
    data_sym = data_ixd.loc[sym]
    return data_sym


def load_data_by_dates(dates: List[int]):

    data = [pd.read_parquet(
        LAGS_FEATURES_TRAINING / 'train_parquet' /
        f'date_id={dates[i]}' / '00000000.parquet' for i in dates)]
    data = pd.concat(data, axis=0)
    return data
