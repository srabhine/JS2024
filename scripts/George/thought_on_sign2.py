"""

@author: Raffaele M Ghigliazza
"""
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from io_lib.paths import LAGS_FEATURES_TRAIN, LAGS_FEATURES_VALID
from math_lib.core import r2_zero
from one_big_lib import stack_features_by_sym, FEATS, TARGET, SYMBOLS
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def load_data(path, start_dt, end_dt):
    data = pl.scan_parquet(path
                           ).select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
        pl.all(),
    ).filter(
        pl.col("date_id").gt(start_dt),
        pl.col("date_id").le(end_dt),
    )

    data = data.collect().to_pandas()

    data.replace([np.inf, -np.inf], 0, inplace=True)
    data = data.fillna(0)
    return data

is_linux = True
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    merged_scaler_df_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_df.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/models/7_base_huberLoss"
    feature_dict_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"

else:
    path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    merged_scaler_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\merged_scalers_df.pkl'
    scaler_std_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\scaler_std_df.pkl'
    feature_dict_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"
    model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\1_0_NN_PlainVanilla\model_save\model_6_perSymbol_scale"


df = load_data(path, start_dt=1490, end_dt=1500)
vld = load_data(path, start_dt=1501, end_dt=1550)


df_by_sym = stack_features_by_sym(df)

sym = 0
df_sym = df_by_sym.swaplevel(axis=1)[sym].ffill()

X = df_sym[FEATS]
y = df_sym[TARGET]

# f, ax = plt.subplots()
# ax.scatter(X['feature_01'], y)
# plt.show()

y_sign = np.sign(y)
# y_pred = np.sign(X['feature_01'])
# accuracy_score(y_sign, y_pred)

y_feat = np.sign(X['feature_01'])

y_pred = np.nan * y.copy()
for i in range(len(y)):
    if i < 1:
        continue
    if y_feat.iloc[i] == y_feat.iloc[i-1]:
        y_pred.iloc[i] = y_sign.iloc[i-1]
    else:
        y_pred.iloc[i] = -y_sign.iloc[i-1]


accuracy_score(y_sign, y_pred.fillna(0))

sg_y = y.std()
acc = pd.Series(0.0, index=SYMBOLS)
r2 = pd.Series(0.0, index=SYMBOLS)
for sym in SYMBOLS:
    print(f'Running {sym}: ', end='')
    df_sym = df_by_sym.swaplevel(axis=1)[sym].ffill()

    X = df_sym[FEATS]
    y = df_sym[TARGET]
    y_feat = np.sign(X['feature_01'])
    y_sign = np.sign(y)
    weights = df_sym['weight']

    y_pred_sign = np.nan * y.copy()
    y_pred = np.nan * y.copy()
    for i in range(len(y)):
        if i < 1:
            continue
        if y_feat.iloc[i] == y_feat.iloc[i-1]:
            y_pred_sign.iloc[i] = y_sign.iloc[i-1]
        else:
            y_pred_sign.iloc[i] = -y_sign.iloc[i-1]
        y_pred.iloc[i] = 0.1 * sg_y * y_pred_sign.iloc[i]

    acc.loc[sym] = accuracy_score(y_sign, y_pred_sign.fillna(0))
    r2[sym] = r2_zero(y, y_pred, weights)
    print(f'{acc.loc[sym]*100:1.2f}%, {r2.loc[sym]:1.4f}')


