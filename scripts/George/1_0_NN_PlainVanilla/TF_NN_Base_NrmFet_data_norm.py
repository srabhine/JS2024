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

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.fillna(0)
    return data


def get_features_classification(file_path: Optional[Any] = None):
    if file_path is None:
        raise ValueError
    feat_types = pd.read_csv(file_path, index_col=0)
    return feat_types.to_dict()['Type']


def get_norm_features_dict(file_path):
    feat_types_dic = get_features_classification(file_path)
    features_to_scale = [feature for feature, ftype in feat_types_dic.items() if ftype == 'normal']
    return features_to_scale



# Function to scale data using loaded scalers
def scale_data_in_place(data, scalers, features_to_scale):
    for symbol_id, group_indices in data.groupby('symbol_id').groups.items():
        if symbol_id in scalers:
            # Retrieve the scaler for this particular symbol_id
            scaler = scalers[symbol_id]

            # Transform the group's "normal" features in place
            data.loc[group_indices, features_to_scale] = scaler.transform(data.loc[group_indices, features_to_scale])

        else:
            # Handle cases where a scaler is missing for a particular symbol_id
            print(f"Warning: No scaler found for symbol_id {symbol_id}. Skipping scaling for this group.")

    return data



is_linux = True
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    scaler_filename = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/all_scalers.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/models/5_base_norm"
    feature_dict_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"

else:
    path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    scaler_filename = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\\all_scalers.pkl'
    feature_dict_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"





# model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\models\\2_base_model_trans_fet"
model_saving_name = "model_5_norm_{epoch:02d}.keras"

feature_names = [f"feature_{i:02d}" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'

features_to_scale = get_norm_features_dict(feature_dict_path)
data_train = load_data(path, start_dt=500, end_dt=1500)
data_valid = load_data(path, start_dt=1501, end_dt=1690)

# Load all scalers from the file


with open(scaler_filename, 'rb') as f:
    loaded_scalers = pickle.load(f)

# Apply scaling to both data_train and data_valid in-place
data_train = scale_data_in_place(data_train, loaded_scalers, features_to_scale)
data_valid = scale_data_in_place(data_valid, loaded_scalers, features_to_scale)



X_train = data_train[feature_names]
y_train = data_train[label_name]
w_train = data_train["weight"]
del data_train

X_valid = data_valid[feature_names]
y_valid = data_valid[label_name]
w_valid = data_valid["weight"]
del data_valid


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
    weighted_mean_true = np.sum(weights * y_true) / np.sum(weights)

    # Calculate the numerator and denominator for R²
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - weighted_mean_true) ** 2)

    # Prevent division by zero
    if denominator == 0:
        return float('nan')

    r2_score = 1 - (numerator / denominator)

    return r2_score

y_pred = model.predict(X_valid)
pred_r2_score = calculate_r2(y_valid, y_pred, w_valid)
print("R2 score: {:.8f}".format(pred_r2_score))
