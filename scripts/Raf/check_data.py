"""

@author: Raffaele M Ghigliazza
"""
import polars as pl
from data_lib.datasets import load_data_by_dates
from io_lib.paths import LAGS_FEATURES_TRAIN

# data = load_data_by_dates(dates=list(range(1654, 1658)))

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()



