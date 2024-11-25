"""

@author: Raffaele M Ghigliazza
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from data_lib.loading import load_scalers, normalize_test_data
from data_lib.variables import IX_IDS_BY_SYM, FEATS
from io_lib.paths import DATA_DIR

# Load scalers
scalers_mu, scalers_sg = load_scalers()
# scalers_mu.to_csv(DATA_DIR / 'scalers_mu_rectangular.csv')
print(scalers_mu.isna().sum().sum())
scalers_mu, scalers_sg = load_scalers(fillna=True)
print(scalers_mu.isna().sum().sum())
# scalers_mu.to_csv(DATA_DIR / 'scalers_mu_rectangular_2.csv')

# Load test data
data_path = DATA_DIR / 'test_part-0.parquet'
test_data = pl.read_parquet(data_path)
test_data = test_data.to_pandas()
np.random.seed(1234)
test_data[FEATS] = test_data[FEATS].fillna(0.0) + \
                   0.001 * np.random.randn(test_data[FEATS].shape[0],
                                   test_data[FEATS].shape[1])

test_norm, scalers_mu, scalers_sg = \
    normalize_test_data(test_data,
                        scalers_mu=scalers_mu,
                        scalers_sg=scalers_sg,
                        fillna=True)

# Check
f, ax = plt.subplots()
ax.scatter(test_data[FEATS[1]].values,
           test_norm[FEATS[1]].values)
plt.show()
