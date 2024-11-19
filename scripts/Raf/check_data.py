"""

@author: Raffaele M Ghigliazza
"""
<<<<<<< HEAD
import polars as pl
from data_lib.datasets import load_data_by_dates
from io_lib.paths import LAGS_FEATURES_TRAINING

# data = load_data_by_dates(dates=list(range(1654, 1658)))

df = pl.scan_parquet(LAGS_FEATURES_TRAINING).collect().to_pandas()
=======
from data_lib.datasets import load_data_by_dates

data = load_data_by_dates(dates=list(range(1654, 1658)))
>>>>>>> origin/main
