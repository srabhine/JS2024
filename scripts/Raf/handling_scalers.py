"""

@author: Raffaele M Ghigliazza
"""

import pandas as pd
import polars as pl

from data_lib.loading import load_scalers
from io_lib.paths import DATA_DIR

# scalers_mu = pd.read_csv(data_path)
# scalers_mu.columns = ['feature', 'symbol_id', 'value']
# scalers_mu.set_index(['symbol_id', 'feature'])
# scalers_mu.swaplevel(axis=0)
# df_tmp = scalers_mu.unstack(level=1)
#
# data_path = DATA_DIR / 'scalers_mu.csv'
# df = pd.read_csv(data_path, header=1,
#                  names=['feature', 'symbol_id', 'value'])
# transformed_df = df.pivot(index='symbol_id', columns='feature',
#                           values='value')
#
# # Display the transformed DataFrame
# transformed_df.head()
scalers_mu, scalers_sg = load_scalers()
# scalers_mu.to_csv(DATA_DIR / 'scalers_mu_rectangular.csv')
print(scalers_mu.isna().sum().sum())
scalers_mu, scalers_sg = load_scalers(fillna=True)
print(scalers_mu.isna().sum().sum())
# scalers_mu.to_csv(DATA_DIR / 'scalers_mu_rectangular_2.csv')

# data_path = DATA_DIR / 'test_part-0.parquet'
# test_data = pl.read_parquet(data_path)
# test_data = test_data.to_pandas()



