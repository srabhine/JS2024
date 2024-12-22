import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Permute, Dense, Add, Activation, Lambda, Layer, Dropout
from sklearn.model_selection import KFold


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def custom_weighted_r2_loss(y_true_with_weights, y_pred):
    # Separate y_true and sample_weight
    y_true = y_true_with_weights[:, 0]
    sample_weight = y_true_with_weights[:, 1]

    # Cast to float32 to avoid type issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sample_weight = tf.cast(sample_weight, tf.float32)

    # Reshape to ensure the tensors are 1D
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    sample_weight = tf.reshape(sample_weight, [-1])

    # Compute numerator and denominator
    numerator = tf.reduce_sum(sample_weight * tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(sample_weight * tf.square(y_true))

    # Avoid division by zero
    denominator = tf.maximum(denominator, tf.keras.backend.epsilon())

    # Compute weighted R² loss
    r2_score = 1 - numerator / denominator
    loss = 1 - r2_score
    return loss


def create_model(input_dim, lr, weight_decay):
    # Create a sequential model
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    # Combine Dense and Swish activation
    model.add(layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Dropout(0.1))  # Assuming dropouts[1] is valid

    model.add(layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Dropout(0.1))  # Assuming dropouts[1] is valid

    model.add(layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(weight_decay)))

    # Output layer with tanh activation
    model.add(layers.Dense(1, activation='tanh'))
    model.add(layers.Lambda(lambda x: 5 * x))

    # Compile model with Mean Squared Error loss
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse',
                  metrics=[tf.keras.metrics.R2Score(class_aggregation='uniform_average')])

    return model

def load_data(path, start_dt, end_dt):
    data = pl.scan_parquet(path).select(
        pl.all(),).filter(
        pl.col("date_id").gt(start_dt),
        pl.col("date_id").le(end_dt),
    ).fill_null(0)

    data = data.collect().to_pandas()

    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data


# def transf_difference(data, features=['feature_12', 'feature_13']):
#     # Iterate over each group of data based on 'symbol_id'
#     for name, sym_df in data.groupby('symbol_id'):
#         # Calculate the difference between the current and the previous row for specified features
#         data.loc[sym_df.index, features] = sym_df[features].diff().astype('float32')
#     return data



is_linux = True
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    training_resp_lag_path = "/home/zt/pyProjects/JaneSt/Team/data/CustomData_1_RespLags/training"
    validation_resp_lag_path = "/home/zt/pyProjects/JaneSt/Team/data/CustomData_1_RespLags/validation"
    # merged_scaler_df_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_df.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/1_HS_Plan/Torch-NN-models"
    # feature_dict_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"

else:
    path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    merged_scaler_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\merged_scalers_df.pkl'
    scaler_std_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\scaler_std_df.pkl'
    feature_dict_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"
    model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\1_0_NN_PlainVanilla\model_save\model_6_perSymbol_scale"



feature_names = [f"feature_{i:02d}" for i in range(79)]
responder_lags = [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'




col_to_train = feature_names + responder_lags

model_saving_name = "NN_0.keras"

X_full = load_data(training_resp_lag_path, start_dt=1200, end_dt=1698)
y_full = X_full[label_name]
w_full = X_full["weight"]
X_train = X_full[col_to_train]


input_dimensions = X_full.shape[1]
lr = 1e-3
weight_decay = 1e-6

# Parameters
N_FOLDS = 5
BATCH_SIZE = 8129
EPOCHS = 100
# Initialize K-Fold
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_index, valid_index) in enumerate(kf.split(X_full)):
    # Split data
    X_train, X_valid = X_full.iloc[train_index], X_full.iloc[valid_index]
    y_train, y_valid = y_full.iloc[train_index], y_full.iloc[valid_index]
    w_train, w_valid = w_full.iloc[train_index], w_full.iloc[valid_index]

    # Initialize model
    model = create_model(input_dimensions, lr, weight_decay)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, mode='min'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{model_saving_path}/NN_{fold}.keras',
            monitor='val_loss', save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6,
                                             mode='min')
    ]

    # Training
    model.fit(
        x=X_train,
        y=y_train,
        sample_weight=w_train,
        validation_data=(X_valid, y_valid, w_valid),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    print(f'Fold-{fold} Training completed.')



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

#
# y_pred = model.predict((X_valid_1, X_valid_2))
#
# pred_r2_score = calculate_r2(y_valid1, np.array(y_pred[0]).flatten(), w_valid)
# print("R2 score: {:.8f}".format(pred_r2_score))
#
#
# model = tf.keras.models.load_model("/home/zt/pyProjects/JaneSt/Team/scripts/George/4_0_NN_MInput/models/NN_Minputs_diff_07.keras")