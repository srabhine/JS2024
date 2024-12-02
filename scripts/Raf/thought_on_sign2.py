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

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
vld = pl.scan_parquet(LAGS_FEATURES_VALID).collect().to_pandas()

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
        # y_pred.iloc[i] = 0.1 * sg_y * y_pred_sign.iloc[i]
        y_pred.iloc[i] = 0.05 * sg_y * y_pred_sign.iloc[i]

    acc.loc[sym] = accuracy_score(y_sign, y_pred_sign.fillna(0))
    r2[sym] = r2_zero(y, y_pred, weights)
    print(f'{acc.loc[sym]*100:1.2f}%, {r2.loc[sym]:1.4f}')


# For submission
def predict(test: pl.DataFrame,
            lags: pl.DataFrame | None) -> pl.DataFrame:
    global old_feature_sign, old_pred_sign

    now_test = test.to_pandas().fillna(0).replace(
        {-np.inf: -1, np.inf: 1})
    features_names = ["symbol_id", "feature_01"]
    symbols = now_test['symbol_id']

    # # Apply the lambda function to compute predictions
    # merged_df = pd.merge(now_test[features_names], before_test[features_names], on = "symbol_id", how="left", suffixes=('_now','_before'))

    if now_test['time_id'].iloc[0] == 0:
        old_pred_sign = pd.Series(1, index=symbols)
        # Change the above to the sign of the last time_id of the previous day
        old_feature_sign = np.sign(now_test['feature_01'])
        pred = 0.078343086 * old_pred_sign

    else:
        new_feature_sign = np.sign(now_test['feature_01'])
        pred = pd.Series(0, index=now_test['symbol_id'])
        for s in now_test['symbol_id']:
            if s in old_feature_sign:
                if new_feature_sign.loc[s] == old_feature_sign.loc[s]:
                    pred.loc[s] = 0.078343086 * old_pred_sign.loc[s]
                else:
                    pred.loc[s] = - 0.078343086 * old_pred_sign.loc[s]
        old_pred_sign = np.sign(pred)
        old_feature_sign = new_feature_sign

    # old_pred = merged_df.apply(sign_compute_lambda, axis=1).fillna(0).values

    predictions = test.select('row_id').with_columns(
        pl.Series(
            name='responder_6',
            values=pred,
            dtype=pl.Float64
        )
    )

    # save the current test a before test, so we can keep it for next round
    before_test = now_test.copy()
    assert len(predictions) == len(test)

    return predictions
