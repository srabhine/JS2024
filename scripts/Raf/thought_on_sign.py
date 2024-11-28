"""

@author: Raffaele M Ghigliazza
"""
import numpy as np
import polars as pl

from io_lib.paths import LAGS_FEATURES_TRAIN, LAGS_FEATURES_VALID
from one_big_lib import stack_features_by_sym

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
vld = pl.scan_parquet(LAGS_FEATURES_VALID).collect().to_pandas()

df_by_sym = stack_features_by_sym(df)

df_by_sym['responder_6']

diffs = np.sign(df_by_sym['responder_6']).diff().dropna()
vals = diffs.values.ravel()
len(vals[vals==0]) / len(vals)


signs = np.sign(df_by_sym['responder_6'])
s_rnd = np.sign(np.random.randn(len(signs))).mean()
sum(abs(signs.mean()) > s_rnd)


