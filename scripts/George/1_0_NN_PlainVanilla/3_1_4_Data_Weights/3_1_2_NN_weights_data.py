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


# def get_features_classification(file_path: Optional[Any] = None):
#     if file_path is None:
#         raise ValueError
#     feat_types = pd.read_csv(file_path, index_col=0)
#     return feat_types.to_dict()['Type']
#
#
# def get_norm_features_dict(file_path):
#     feat_types_dic = get_features_classification(file_path)
#     features_to_scale = [feature for feature, ftype in feat_types_dic.items() if ftype == 'normal']
#     return features_to_scale



# Function to scale data using loaded scalers



def reconstruct_scalers(scalers_df: pd.DataFrame, all_features: List[str]) -> pd.DataFrame:

    # Create a complete scaler DataFrame with defaults
    complete_scalers_df = pd.DataFrame(index=all_features, columns=['mean', 'std'])

    # Fill in existing scalers
    for feature in scalers_df.index:
        complete_scalers_df.loc[feature, 'mean'] = scalers_df.at[feature, 'mean']
        complete_scalers_df.loc[feature, 'std'] = scalers_df.at[feature, 'std']

    # Fill missing scalers with default values
    complete_scalers_df = complete_scalers_df.fillna({'mean': 0, 'std': 1})
    scalers_mi = complete_scalers_df.stack().unstack(level=0)

    return scalers_mi





def apply_scalers_with_multiindex(valid_data: pd.DataFrame, scalers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the validation dataset using MultiIndex and vectorized operations.

    Parameters:
    - valid_data: pd.DataFrame, Validation data to be normalized.
    - scalers_df: pd.DataFrame, Complete scalers DataFrame with mean and std.

    Returns:
    - Scaled validation DataFrame.
    """
    # Ensure valid_data columns are aligned with scalers
    data_train_mi = data_train.reindex(columns=scalers_df.index)

    # Scale data using broadcasting
    scaled_data = (data_train_mi - scalers_df.loc['mean']) / scalers_df.loc['std']

    return scaled_data


def transform_data_with_scalers(data: pd.DataFrame, scalers_df: pd.DataFrame) -> pd.DataFrame:
    def reconstruct_scalers_for_data(data: pd.DataFrame, scalers_df: pd.DataFrame) -> pd.DataFrame:
        # Create missing features set to determine if any features are not in scalers_df
        missing_features = set(data.columns) - set(scalers_df.index)

        # Create a new DataFrame for the complete scalers with defaults for missing features
        complete_scalers_df = scalers_df.copy()

        # Add missing features with default mean=0 and std=1
        for feature in missing_features:
            complete_scalers_df.loc[feature] = {'mean': 0, 'std': 1}

        # Ensure correct order aligned with data columns
        complete_scalers_df = complete_scalers_df.reindex(data.columns)

        return complete_scalers_df
    # First, ensure scalers_df is reconstructed to include all data features
    scalers_df = reconstruct_scalers_for_data(data, scalers_df)

    # Perform scaling using vectorized broadcasting
    scaled_data = (data - scalers_df['mean']) / scalers_df['std']

    return scaled_data




is_linux = True
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    scaler_filename = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_whole.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/models/5_base_norm"
    feature_dict_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"

else:
    path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    scaler_filename = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\\all_scalers.pkl'
    feature_dict_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"


features_to_scale = ['feature_01', 'feature_04','feature_18','feature_19','feature_33','feature_36','feature_39','feature_40',
                     'feature_41','feature_42','feature_43', 'feature_44','feature_45','feature_46','feature_50','feature_51',
                     'feature_52','feature_53','feature_54','feature_55','feature_56','feature_57','feature_63','feature_64',
                     'feature_78']


# model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\models\\2_base_model_trans_fet"
model_saving_name = "model_6_weightsSel_{epoch:02d}.keras"

feature_names = [f"feature_{i:02d}" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'

# features_to_scale = get_norm_features_dict(feature_dict_path)
data_train = load_data(path, start_dt=600, end_dt=1500)
data_valid = load_data(path, start_dt=1501, end_dt=1690)

data_train = data_train[data_train['weight']>=1.35]
data_valid = data_valid[data_valid['weight']>=1.35]



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
