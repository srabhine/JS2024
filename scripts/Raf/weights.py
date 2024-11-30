"""

@author: Raffaele M Ghigliazza
"""

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from io_lib.paths import LAGS_FEATURES_TRAIN, LAGS_FEATURES_VALID
from math_lib.core import r2_zero
from one_big_lib import stack_features_by_sym
from plot_lib.core import PARAMS_HELPER

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
vld = pl.scan_parquet(LAGS_FEATURES_VALID).collect().to_pandas()

weights = df['weight'].copy()

weights.sort_values(ascending=True, inplace=True)

f, axs = plt.subplots(2, figsize=(8, 6))
axs[0].plot(weights.values)
axs[1].plot(weights.values.cumsum())
plt.show()

hist, bin_edges = np.histogram(weights, 100)
xc = (bin_edges[:-1] + bin_edges[1:]) / 2
hist_w = hist * xc

f, axs = plt.subplots(3, figsize=(8, 9))
axs[0].bar(xc, hist, width=0.05)
axs[1].bar(xc, hist_w, width=0.05)
axs[2].plot(xc, hist_w.cumsum())
axs[2].axhline(hist_w.sum()/2, **PARAMS_HELPER)
axs[2].axhline(hist_w.sum()*0.1, **PARAMS_HELPER)
axs[2].axhline(hist_w.sum()*0.2, **PARAMS_HELPER)
axs[2].axvline(2.25, **PARAMS_HELPER)
axs[2].axvline(1.10, **PARAMS_HELPER)
axs[2].axvline(1.35, **PARAMS_HELPER)
plt.show()

len(weights[weights>1.35]) / len(weights)

weights_ = df['weight']
y_true_ = df['responder_6']

y_pred_ = 0 * np.ones_like(y_true_)


r2w = r2_zero(y_true_, y_pred_, weights_)
print(r2w)

y_pred_ = y_true_.mean()
r2w = r2_zero(y_true_, y_pred_, weights_)
print(r2w)






