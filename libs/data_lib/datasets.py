"""

@author: Raffaele M Ghigliazza
"""
import pandas as pd

from io_lib.paths import DATA_DIR


def get_indexed_dataset():
    data = pd.read_parquet(DATA_DIR / 'part-0_id_9.parquet',
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

