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


y = df['responder_6']
y_s = np.sign(y)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

X_num = df.select_dtypes(exclude='object')
X_cat = df.select_dtypes(include='object')
encoder = OneHotEncoder(handle_unknown='error')
X_encoded = encoder.fit_transform(y_s.to_frame())
print(X_encoded)

classes = [-1, 0, 1]
enc = LabelEncoder()
encoder.fit(np.array(classes).reshape(-1, 1))

X_encoded = encoder.transform(y_s.values.reshape(-1, 1))

a = X_encoded.todense()
print(a)