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





# def normalize_data(data, merged_scaler_df):
#     feature_names = [f"feature_{i:02d}" for i in range(79)]
#     feature_names_mean = [f"feature_{i:02d}_mean" for i in range(79)]
#     feature_names_std = [f"feature_{i:02d}_std" for i in range(79)]
#
#     merged_data = pd.merge(data, merged_scaler_df, on='symbol_id', how='left')
#     means_array = merged_data[feature_names_mean].values
#     stds_array = merged_data[feature_names_std].values
#     normalized_features = (merged_data[feature_names].values - means_array) / stds_array
#     return normalized_features


def normalize_data(data, merged_scaler_df):
    # Define feature names
    feature_names = [f"feature_{i:02d}" for i in range(79)]
    feature_names_mean = [f"feature_{i:02d}_mean" for i in range(79)]
    feature_names_std = [f"feature_{i:02d}_std" for i in range(79)]

    # Set 'symbol_id' as the index for quick lookup
    data.set_index('symbol_id', inplace=True)
    # merged_scaler_df.set_index('symbol_id', inplace=True)

    # Dictionary to hold features, weights, and targets for each symbol_id
    data_dict = {}

    # Iterate over the unique symbol_ids
    for symbol_id in range(0,39):
        # Select feature data for the current symbol_id
        feature_values = data.loc[symbol_id, feature_names].values

        # Select mean and std values for the current symbol_id
        means = merged_scaler_df.loc[symbol_id, feature_names_mean].values
        stds = merged_scaler_df.loc[symbol_id, feature_names_std].values

        # Normalize features using broadcasting
        normalized_features = (feature_values - means) / stds

        # Extract weights and target values
        weights = data.loc[symbol_id, 'weight'].values
        target = data.loc[symbol_id, 'responder_6'].values

        # Store in dictionary
        data_dict[symbol_id] = {
            'features': normalized_features,
            'weights': weights,
            'target': target
        }

    return data_dict






is_linux = True
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    merged_scaler_df_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_df.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/models/5_base_norm"
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


model_saving_name = "model_5_norm_{epoch:02d}.keras"

feature_names = [f"feature_{i:02d}" for i in range(79)]
feature_names_mean = [f"feature_{i:02d}_mean" for i in range(79)]
feature_names_std = [f"feature_{i:02d}_std" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'

# features_to_scale = get_norm_features_dict(feature_dict_path)
data_train = load_data(path, start_dt=600, end_dt=1500)
data_valid = load_data(path, start_dt=1501, end_dt=1690)



with open(merged_scaler_df_path, 'rb') as f:
    merged_scaler_df = pickle.load(f)
    



# X_train = data_train[feature_names]


train_dict = normalize_data(data_train, merged_scaler_df)
del data_train

# X_valid = data_valid[feature_names]
valid_dict = normalize_data(data_valid, merged_scaler_df)
del data_valid




lr = 0.01
weight_decay = 1e-6
input_dimensions = train_dict[0]['features'].shape[1]
model = create_model(input_dimensions, lr, weight_decay)

ca = [
    tf.keras.callbacks.EarlyStopping(monitor='val_r2_score', patience=30, mode='max'),
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

symbol_id = 38 # <<<<<<<<<<<<<< Select symbol to train

model.fit(
    x=train_dict[symbol_id]['features'],  # Input features for training
    y=train_dict[symbol_id]['target'],  # Target labels for training
    sample_weight=train_dict[symbol_id]['weights'],  # Sample weights for training
    validation_data=(valid_dict[symbol_id]['features'], valid_dict[symbol_id]['target'], valid_dict[symbol_id]['weights']),  # Validation data
    batch_size=8029,  # Batch size
    epochs=100,  # Number of epochs
    callbacks=ca,  # Callbacks list, if any
    verbose=1,  # Verbose output during training
    shuffle=False
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
