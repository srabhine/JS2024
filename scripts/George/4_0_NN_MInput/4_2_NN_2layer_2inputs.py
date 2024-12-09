import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Permute, Dense, Add, Activation, Lambda, Layer, \
    Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def create_model(input_dimensions1, input_dimensions2, lr, weight_decay):
    input_group_1 = Input(shape=(input_dimensions1,), name='input_1')
    input_group_2 = Input(shape=(input_dimensions2,), name='input_2')

    layer2 = Dense(128, activation='swish')(input_group_2)
    layer2 = Dropout(0.1)(layer2)
    layer2 = Dense(32, activation='swish', kernel_regularizer=regularizers.l2(weight_decay))(layer2)
    layer2 = Dropout(0.005)(layer2)
    layer2 = Dense(1, activation='tanh', name="responder_6_return")(
        layer2)

    input_to_ly1 = Concatenate()([layer2, input_group_1])
    layer1 = Dense(512, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(input_to_ly1)
    layer1 = Dropout(0.1)(layer1)
    layer1 = Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(input_to_ly1)
    layer1 = Dropout(0.1)(layer1)
    layer1 = Dense(128, activation='swish', kernel_regularizer=regularizers.l2(1e-5))(layer1)
    layer1 = Dropout(0.1)(layer1)
    layer1 = Dense(32, activation='swish', kernel_regularizer=regularizers.l2(1e-6))(layer1)
    layer1 = Dropout(0.1)(layer1)
    output = Dense(1, activation='linear', name="responder_6")(layer1)

    # Create the model
    model = Model(inputs=[input_group_1, input_group_2], outputs=[output, layer2])

    # Compile the model
    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss={"responder_6": "mse", "responder_6_return": "mse"},
        metrics={"responder_6": tf.keras.metrics.R2Score(),
                 "responder_6_return": tf.keras.metrics.R2Score()}
    )

    model.summary()
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


# def transf_difference(data, features=['feature_12', 'feature_13']):
#     # Iterate over each group of data based on 'symbol_id'
#     for name, sym_df in data.groupby('symbol_id'):
#         # Calculate the difference between the current and the previous row for specified features
#         data.loc[sym_df.index, features] = sym_df[features].diff().astype('float32')
#     return data

def transf_difference(data, features=['feature_12', 'feature_13']):
    # Iterate over each group of data based on 'symbol_id'
    for name, sym_df in data.groupby('symbol_id'):
        # Calculate the difference between the current and the previous row for specified features
        for feature in features:
            # Create a new column name for the difference
            new_column_name = f"{feature}_dif"
            # Calculate the difference and assign it to the new column
            data.loc[sym_df.index, new_column_name] = sym_df[feature].diff().astype('float32')
    return data

def fill_na_inf(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        # For pandas objects, use fillna and replace
        data = data.fillna(0)
        data = data.replace([np.inf, -np.inf], 0)
    elif isinstance(data, np.ndarray):
        # For numpy arrays, use np.isnan and np.isinf
        inf_mask = np.isinf(data)
        nan_mask = np.isnan(data)
        data[inf_mask] = 0
        data[nan_mask] = 0
    else:
        raise TypeError("Input must be a numpy array, pandas DataFrame, or pandas Series.")
    return data


is_linux = True
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    merged_scaler_df_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/0_1_Transform_and_save_Data/temp_scalers/scalers_df.pkl"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/4_0_NN_MInput/models"
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



normal_features = ['feature_01', 'feature_04', 'feature_18', 'feature_19', 'feature_33',
                   'feature_36', 'feature_39', 'feature_40', 'feature_41', 'feature_42',
                   'feature_43', 'feature_44', 'feature_45', 'feature_46', 'feature_50',
                   'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55',
                   'feature_56', 'feature_57', 'feature_63', 'feature_64', 'feature_78']
normal_features_dif = [feature + '_dif' for feature in normal_features]

integrated_features =  ['feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24',
       'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29',
       'feature_30', 'feature_31']
integrated_features_dif = [feature + '_dif' for feature in integrated_features]

fat_features = ['feature_05', 'feature_06', 'feature_07', 'feature_08', 'feature_15',
       'feature_37', 'feature_38', 'feature_47', 'feature_48', 'feature_49',
       'feature_58', 'feature_59', 'feature_60', 'feature_62', 'feature_65',
       'feature_66']
fat_features_dif = [feature + '_dif' for feature in fat_features]

cyclic_features =  ['feature_05', 'feature_06', 'feature_07', 'feature_08', 'feature_15',
       'feature_37', 'feature_38', 'feature_47', 'feature_48', 'feature_49',
       'feature_58', 'feature_59', 'feature_60', 'feature_62', 'feature_65',
       'feature_66']

cyclic_features_dif = [feature + '_dif' for feature in cyclic_features]


features_to_transform = normal_features + integrated_features+ cyclic_features

col_to_train1 = ['symbol_id', 'time_id'] + feature_names
col_to_train2 = ['symbol_id', 'time_id'] + normal_features_dif + integrated_features_dif + cyclic_features_dif

model_saving_name = "NN_Minputs_diff_{epoch:02d}.keras"

X_train = load_data(path, start_dt=1200, end_dt=1500)
y_train1 = X_train["responder_6"]
y_train2 = transf_difference(X_train, features=["responder_6"])['responder_6_dif'].fillna(0).values
w_train = X_train["weight"]
X_train_1 = X_train[col_to_train1]
X_train_2 = transf_difference(X_train, features=features_to_transform)[col_to_train2]


X_valid = load_data(path, start_dt=1501, end_dt=1690)
y_valid1 = X_valid["responder_6"]
y_valid2 = transf_difference(X_valid, features=["responder_6"])['responder_6_dif'].fillna(0).values
w_valid = X_valid["weight"]
X_valid_1 = X_valid[col_to_train1]
X_valid_2 = transf_difference(X_valid, features=features_to_transform)[col_to_train2]




X_train_1 = fill_na_inf(X_train_1)
X_train_2 = fill_na_inf(X_train_2)
y_train1  = fill_na_inf(y_train1)
y_train2  = fill_na_inf(y_train2)
w_train   = fill_na_inf(w_train)




X_valid_1 = fill_na_inf(X_valid_1)
X_valid_2 = fill_na_inf(X_valid_2)
y_valid1  = fill_na_inf(y_valid1)
y_valid2  = fill_na_inf(y_valid2)
w_valid   = fill_na_inf(w_valid)



lr = 0.01
weight_decay = 1e-6
input_dimensions = X_train.shape[1]
model = create_model(input_dimensions1=X_train_1.shape[1],
                     input_dimensions2=X_train_2.shape[1],
                     lr=0.001,
                     weight_decay=0.0001)



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
    x=[X_train_1, X_train_2],
    y={"responder_6": y_train1, "responder_6_return": y_train2},
    sample_weight=w_train,
    validation_data=(
        [X_valid_1, X_valid_2],
        {"responder_6": y_valid1, "responder_6_return": y_valid2},
        w_valid
    ),
    batch_size=8029,
    epochs=100,
    callbacks=ca,
    verbose=1,
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


y_pred = model.predict((X_valid_1, X_valid_2))

pred_r2_score = calculate_r2(y_valid1, np.array(y_pred[0]).flatten(), w_valid)
print("R2 score: {:.8f}".format(pred_r2_score))


model = tf.keras.models.load_model("/home/zt/pyProjects/JaneSt/Team/scripts/George/4_0_NN_MInput/models/NN_Minputs_diff_07.keras")