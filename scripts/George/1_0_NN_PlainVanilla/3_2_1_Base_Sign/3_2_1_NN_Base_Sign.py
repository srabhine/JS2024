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

def transform_sign(data):
    temp = np.sign(data[sign_features])
    temp.columns = [c + "_sign" for c in temp.columns]
    data = pd.concat([data, temp], axis=1)
    return data


is_linux = True
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    merged_scaler_df_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_df.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/1_0_NN_PlainVanilla/3_2_1_Base_Sign/models"
    feature_dict_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"

else:
    path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    merged_scaler_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\merged_scalers_df.pkl'
    scaler_std_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\scaler_std_df.pkl'
    feature_dict_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"
    model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\1_0_NN_PlainVanilla\model_save\model_6_perSymbol_scale"


features_to_scale = ['feature_01', 'feature_04','feature_18','feature_19','feature_33','feature_36','feature_39','feature_40',
                     'feature_41','feature_42','feature_43', 'feature_44','feature_45','feature_46','feature_50','feature_51',
                     'feature_52','feature_53','feature_54','feature_55','feature_56','feature_57','feature_63','feature_64',
                     'feature_78']


model_saving_name = "model_sign_{epoch:02d}.keras"

feature_names = [f"feature_{i:02d}" for i in range(79)]
feature_names_mean = [f"feature_{i:02d}_mean" for i in range(79)]
feature_names_std = [f"feature_{i:02d}_std" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'

col_to_train = ['symbol_id', 'time_id'] + feature_names

sign_features = ['feature_20', 'feature_21', 'feature_22', 'feature_23',
       'feature_24', 'feature_25', 'feature_26', 'feature_27',
       'feature_28', 'feature_29', 'feature_30', 'feature_31']


sign_features_post = [i+"_sign" for i in sign_features]





X_train = load_data(path, start_dt=1200, end_dt=1500)
X_train = X_train.fillna(0)
y_train = X_train[label_name]
w_train = X_train["weight"]
# X_train = transf_moving_avg(X_train, days=10, features=norm_features)
X_train = transform_sign(X_train)
X_train = X_train[col_to_train+ sign_features_post]



X_valid = load_data(path, start_dt=1501, end_dt=1690)
X_valid = X_valid.fillna(0)
y_valid = X_valid[label_name]
w_valid = X_valid["weight"]
# X_valid = transf_moving_avg(X_valid, days=10, features=norm_features)
X_valid = transform_sign(X_valid)
X_valid = X_valid[col_to_train + sign_features_post]


X_train = X_train.fillna(0)
w_train = w_train.fillna(0)
X_train = X_train.replace([np.inf, -np.inf], 0)


X_valid = X_valid.fillna(0)
w_valid = w_valid.fillna(0)
X_valid = X_valid.replace([np.inf, -np.inf], 0)





lr = 0.01
weight_decay = 1e-6
input_dimensions = X_train.shape[1]
model = create_model(input_dimensions, lr, weight_decay)

ca = [
    tf.keras.callbacks.EarlyStopping(monitor='val_r2_score', patience=25, mode='max'),
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
