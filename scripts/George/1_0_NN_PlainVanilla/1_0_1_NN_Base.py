import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import Optional, List, Union, Dict, Any
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
import random

def set_random_seeds(seed=42):
    # Set the random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Before creating and training your model, call the function
set_random_seeds(42)


def create_model(input_dim, lr, weight_decay):
    # Create a sequential model
    model = models.Sequential()

    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))  # Assuming dropouts[1] is valid
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))  # Assuming dropouts[1] is valid
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(weight_decay)))
    # Output layer
    model.add(layers.Dense(1, activation='tanh'))

    # Compile model with Mean Squared Error loss
    # model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=[WeightedR2()])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=[tf.keras.metrics.R2Score(class_aggregation='uniform_average')])
    return model


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



feature_names = [f"feature_{i:02d}" for i in range(79)]
feature_names_mean = [f"feature_{i:02d}_mean" for i in range(79)]
feature_names_std = [f"feature_{i:02d}_std" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'



# =================Needs Edit=======================================
# col_to_train = ['symbol_id', 'date_id', 'time_id'] + feature_names
model_saving_name = "model_0_base_{epoch:02d}.keras"
col_to_train = feature_names

with open(merged_scaler_df_path, 'rb') as f:
    merged_scaler_df = pickle.load(f)

# X_train = data_train[feature_names]
X_train = load_data(path, start_dt=1200, end_dt=1500)
# X_train = data_train[feature_names]
y_train = X_train[label_name]
w_train = X_train["weight"]
X_train = X_train[col_to_train]
# del data_train

X_valid = load_data(path, start_dt=1501, end_dt=1690)
# X_valid = data_valid[feature_names]
y_valid = X_valid[label_name]
w_valid = X_valid["weight"]
X_valid = X_valid[col_to_train]
# del data_valid


lr = 0.01
weight_decay = 1e-6
input_dimensions = X_train.shape[1]
model = create_model(input_dimensions, lr, weight_decay)

ca = [
    tf.keras.callbacks.EarlyStopping(monitor='val_r2_score', patience=50, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{model_saving_path}/{model_saving_name}',
        monitor='val_loss', save_best_only=False),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Metric to be monitored
        factor=0.1,  # Factor by which the learning rate will be reduced
        patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1,  # Verbosity mode
        min_lr=1e-6  # Lower bound on the learning rate
    )

]

model.fit(
    x=X_train,  # Input features for training
    y=y_train,  # Target labels for training
    sample_weight=w_train,  # Sample weights for training
    validation_data=(X_valid, y_valid, w_valid),  # Validation data
    batch_size=8029,  # Batch size
    epochs=100,  # Number of epochs
    callbacks=ca,  # Callbacks list, if any
    verbose=1,  # Verbose output during training
    shuffle=True
)


def calculate_r2(y_true, y_pred, weights):
    # Convert inputs to numpy arrays and check their shapes
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    weights = np.asarray(weights).flatten()

    if not (y_true.shape == y_pred.shape == weights.shape):
        raise ValueError(
            f'Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}, weights {weights.shape}'
        )

    # Calculate weighted mean of y_true
    # weighted_mean_true = np.sum(weights * y_true) / np.sum(weights)

    # Calculate the numerator and denominator for R²
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true) ** 2)

    # Prevent division by zero
    if denominator == 0:
        return float('nan')

    r2_score = 1 - (numerator / denominator)

    return r2_score


y_pred = model.predict(X_valid)
pred_r2_score = calculate_r2(y_valid, y_pred, w_valid)
print("R2 score: {:.8f}".format(pred_r2_score))
