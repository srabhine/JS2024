import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks

from io_lib.paths import LAGS_FEATURES_TRAINING, \
    LAGS_FEATURES_VALIDATION


def create_model(input_dim, lr, weight_decay):
    # Create a sequential model
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Add BatchNorm, ELU, Dropout, and Dense layers
    # BatchNormalization over the feature dimension (default for dense)
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(weight_decay)))

    # Output layer
    model.add(layers.Dense(1, activation='tanh'))
    # model.add(layers.Dense(1, activation='linear'))

    # Compile model with Mean Squared Error loss
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=[tf.keras.metrics.R2Score(class_aggregation='uniform_average')])

    return model

lr = 0.01
weight_decay = 5e-4

input_dim = 167
model = create_model(input_dim=input_dim, lr = lr, weight_decay=weight_decay)

model.summary()
path = (r'C:\Users\rghig\Dropbox\Kaggle\JaneS\JS2024\data'
        r'\lags_features\models\tf_nn_model10_batch.keras')
model.load_weights(path)

feature_names = [f"feature_{i:02d}" for i in range(79)] + [f"feature_{i:02d}_lag_1" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAINING).collect().to_pandas()
valid = pl.scan_parquet(LAGS_FEATURES_VALIDATION).collect().to_pandas()


# Select subset
sym = 1
df_sym = df[df['symbol_id'] == sym]
valid_sym = valid[valid['symbol_id'] == sym]
print(df_sym.head())

feature_names = [f"feature_{i:02d}" for i in range(79)] + [f"feature_{i:02d}_lag_1" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'
# df = pd.concat([df, valid]).reset_index(drop=True)
df_sym[feature_names] = df_sym[feature_names].ffill().fillna(0)
valid_sym[feature_names] = valid_sym[feature_names].ffill().fillna(0)


X_train = df_sym[ feature_names ]
y_train = df_sym[ label_name ]
w_train = df_sym[ "weight" ]
X_valid = valid_sym[ feature_names ]
y_valid = valid_sym[ label_name ]
w_valid = valid_sym[ "weight" ]

X_new = X_valid.to_numpy()  # Replace with actual data
predictions = model.predict(X_new)


f, axs = plt.subplots(3, figsize=(8, 8),
                      sharey=True)
axs[0].plot(y_valid.values / y_valid.std())
axs[1].plot(predictions / predictions.std())
axs[2].scatter(y_valid.values/ y_valid.std(),
               predictions/ predictions.std())
plt.show()


f, ax = plt.subplots(1)
ax.plot(predictions)
plt.show()

