import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import (layers, models, optimizers,
                              regularizers, callbacks)

from io_lib.paths import LAGS_FEATURES_TRAIN, \
    LAGS_FEATURES_VALID, MODELS_DIR
from models_lib.dnns import dnn_model


# Data
feature_names = ([f"feature_{i:02d}" for i in range(79)]
                 + [f"feature_{i:02d}_lag_1" for i in range(79)]
                 + [f"responder_{idx}_lag_1" for idx in range(9)])
label_name = 'responder_6'
weight_name = 'weight'

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
vld = pl.scan_parquet(LAGS_FEATURES_VALID).collect().to_pandas()

# Select subset
sym = 1
df_sym = df[df['symbol_id'] == sym]
vld_sym = vld[vld['symbol_id'] == sym]
print(df_sym.head())

# df = pd.concat([df, vld]).reset_index(drop=True)
df_sym[feature_names] = df_sym[feature_names].ffill().fillna(0)
vld_sym[feature_names] = vld_sym[feature_names].ffill().fillna(0)

X_train = df_sym[ feature_names ]
y_train = df_sym[ label_name ]
w_train = df_sym[ "weight" ]
X_vld = vld_sym[ feature_names ]
y_vld = vld_sym[ label_name ]
w_vld = vld_sym[ "weight" ]


lr = 0.01
weight_decay = 5e-4
input_dim = df_sym[feature_names].shape[1]
X_new = X_vld.to_numpy()  # Replace with actual data
predictions = []
for out_layer in ['tanh', 'linear']:
    path = str(MODELS_DIR) + f'/tf_nn_model10_batch_{out_layer}.keras'

    model = dnn_model(input_dim=input_dim, lr=lr,
                      weight_decay=weight_decay,
                      out_layer=out_layer,
                      simplified=True)

    model.summary()
    model.load_weights(path)
    pred_tmp = model.predict(X_new)
    predictions.append(pred_tmp)

    f, ax = plt.subplots(1)
    ax.plot(pred_tmp)
    plt.show()

    f, axs = plt.subplots(3, figsize=(8, 8),
                          sharey=True)
    axs[0].plot(y_vld.values)
    axs[1].plot(pred_tmp)
    axs[2].scatter(y_vld.values, pred_tmp)
    plt.show()

predictions = np.column_stack(predictions)

f, ax = plt.subplots(1, sharex=True, sharey=True)
ax.scatter(predictions[:, 0], predictions[:, 1])
ax.set(xlabel='tanh', ylabel='linear')
ax.set(xlim=(-1, 1), ylim=(-1, 1))
plt.show()

# f, axs = plt.subplots(3, figsize=(8, 8),
#                       sharey=True)
# axs[0].plot(y_vld.values / y_vld.std())
# axs[1].plot(predictions / predictions.std())
# axs[2].scatter(y_vld.values/ y_vld.std(),
#                predictions/ predictions.std())
# plt.show()

