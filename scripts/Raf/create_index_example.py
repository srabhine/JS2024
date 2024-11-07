import pandas as pd

from io_lib.paths import DATA_DIR

# Load data
data = pd.read_parquet(DATA_DIR / 'part-0_id_9.parquet',
                       engine='pyarrow')

# Group
data_ixd = data.copy()
df_tmp = data_ixd[['date_id', 'time_id']].drop_duplicates()
df_tmp['index_id'] = range(len(df_tmp))
df_tmp.set_index(['date_id', 'time_id'], inplace=True)
data_ixd['index_id'] = 0  # create the extra column

# This takes time (not efficient)
for i in data_ixd.index:
    print(f'\r{i}', end='')
    d_i, t_i = data_ixd.loc[i, 'date_id'], data_ixd.loc[i, 'time_id']
    data_ixd.loc[i, 'index_id'] = df_tmp.loc[(d_i, t_i), 'index_id']

# Use the new column as index
data_ixd.set_index(['symbol_id', 'index_id'],
                   inplace=True)
data_ixd.sort_index(axis=0, inplace=True)

# Save
# data_ixd.to_csv(DATA_DIR / 'data_ixd.parquet')
# This took quite long time

