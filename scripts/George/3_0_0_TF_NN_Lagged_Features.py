"""

@authors: George, Raffaele
"""
import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from data_lib.core_tf import prepare_dataset
from io_lib.paths import LAGS_FEATURES_TRAIN, \
    LAGS_FEATURES_VALID, MODELS_DIR
from models_lib.dnns import dnn_model

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
df_sym = df[df['symbol_id'] == sym].copy()
vld_sym = vld[vld['symbol_id'] == sym].copy()
print(df_sym.head())

df_sym[feature_names] = df_sym[feature_names].ffill().fillna(0)
vld_sym[feature_names] = vld_sym[feature_names].ffill().fillna(0)


# Prepare datasets
X_train = df_sym[feature_names]
y_train = df_sym[label_name]
w_train = df_sym["weight"]
X_valid = vld_sym[feature_names]
y_valid = vld_sym[label_name]
w_valid = vld_sym["weight"]

# Set seed
# seed = 0
seed = 1234
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)


train_dataset = prepare_dataset(df_sym, w_train, feature_names,
                                label_name, batch_size=8129)
valid_dataset = prepare_dataset(vld_sym, w_valid, feature_names,
                                label_name, batch_size=8129)


lr = 0.01
weight_decay = 5e-4
# weight_decay = 0

input_dim = df_sym[feature_names].shape[1]
# out_layer = 'tanh'
out_layer = 'linear'

model = dnn_model(input_dim=input_dim, lr=lr,
                  weight_decay=weight_decay,
                  out_layer=out_layer,
                  simplified=True)
model.summary()

path = str(MODELS_DIR) + f'/tf_nn_model10_batch_{out_layer}.keras'

ca = [
    tf.keras.callbacks.EarlyStopping(monitor='val_r2_score',
                                     patience=15, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_r2_score',  # Metric to be monitored
        factor=0.1,  # Factor by which the learning rate will be reduced
        patience=8,  # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1,  # Verbosity mode
        min_lr=1e-6  # Lower bound on the learning rate
    )

]

model.fit(
    train_dataset.map(lambda x, y, w: (x, y, {'sample_weight': w})),
    epochs=70,
    validation_data=valid_dataset.map(lambda x, y, w: (x, y, {'sample_weight': w})),
    callbacks=ca
)


model.save(path)


# Assume X_new is your new data you want to make predictions on
# This should be a NumPy array or a Tensor with shape (num_samples, num_features)
X_new = X_valid.to_numpy()  # Replace with actual data

# Make predictions
predictions = model.predict(X_new)
r2_metric = tf.keras.metrics.R2Score(class_aggregation='uniform_average')

if not isinstance(y_valid, np.ndarray):
    y_valid = y_valid.to_numpy()  # Convert to numpy array if it is a DataFrame

r2_metric.update_state(y_true=y_valid, y_pred=predictions)
r2_score_value = r2_metric.result().numpy()

print(f"R² Score on validation data: {r2_score_value:1.6f}")
