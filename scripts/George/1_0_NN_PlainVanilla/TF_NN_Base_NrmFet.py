import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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

def load_data(train_path,start_id,end_id):
    # df = pl.scan_parquet(f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/FOLD3").collect().to_pandas()
    folder_paths = [
        f"{train_path}/train_parquet_{partition_id}.parquet"
        for partition_id in range(start_id, end_id + 1)
    ]
    lazy_frames = [pl.scan_parquet(path) for path in folder_paths]
    combined_lazy_df = pl.concat(lazy_frames)

    data = combined_lazy_df.collect().to_pandas()

    data[feature_names] = data[feature_names].ffill().fillna(0)
    return data



train_path = "/home/zt/pyProjects/JaneSt/Team/data/transformed_data"
model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/models/2_base_model_trans_fet"
model_saving_name = "model_2_Base_transFet_{epoch:02d}.keras"

feature_names = [f"feature_{i:02d}" for i in range(79)]
label_name = 'responder_6'
weight_name = 'weight'


data_train = load_data(train_path,start_id=4,end_id=7)
data_valid = load_data(train_path,start_id=8,end_id=9)


X_train = data_train[ feature_names ]
y_train = data_train[ label_name    ]
w_train = data_train[ "weight"      ]
X_valid = data_valid[ feature_names ]
y_valid = data_valid[ label_name    ]
w_valid = data_valid[ "weight"      ]

del data_train
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
    y=y_train,                          # Target labels for training
    sample_weight=w_train,              # Sample weights for training
    validation_data=(X_valid, y_valid, w_valid),  # Validation data
    batch_size=8092,                      # Batch size
    epochs=100,                        # Number of epochs
    callbacks=ca,                # Callbacks list, if any
    verbose=1,                           # Verbose output during training
    shuffle=True
)

