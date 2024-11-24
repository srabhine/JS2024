import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def create_lstm_model(input_shape, lr, weight_decay):
    model = Sequential()
    model.add(
        LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='swish', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(1, activation='tanh'))  # Ensure single output
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def load_data(train_path, valid_path):
    # df = pl.scan_parquet(f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/FOLD3").collect().to_pandas()
    start_id = 1500
    end_id = 1646
    folder_paths = [
        f"{train_path}/date_id={date_id}/*.parquet"
        for date_id in range(start_id, end_id + 1)
    ]
    lazy_frames = [pl.scan_parquet(path) for path in folder_paths]
    combined_lazy_df = pl.concat(lazy_frames)

    data_train = combined_lazy_df.collect().to_pandas()
    data_valid = pl.scan_parquet(valid_path).collect().to_pandas()

    data_train[feature_names] = data_train[feature_names].ffill().fillna(0)
    data_valid[feature_names] = data_valid[feature_names].ffill().fillna(0)
    return data_train, data_valid




def create_sequences(data, feature_names, target_name, n_steps):
    sequences = []
    labels = []

    for i in tqdm(range(len(data) - n_steps)):
        # Take segment of the data for the sequence
        sequence = data.iloc[i:i + n_steps][feature_names].values
        # Target the next time step
        label = data.iloc[i + n_steps][target_name]
        sequences.append(sequence)
        labels.append(label)

    return np.array(sequences), np.array(labels)

train_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/NN_Models/data/Data_date_id_Partition/training.parquet"
valid_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/NN_Models/data/Data_date_id_Partition/validation.parquet"
model_saving_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/NN_Models/models/model_1_multi_group"
model_saving_name = "model_1_LSTM_{epoch:02d}.keras"

feature_names = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
target_name   = 'responder_6'
weight_name   = 'weight'
n_steps       = 5

data_train, data_valid = load_data(train_path, valid_path)

X_train = data_train[ feature_names ]
y_train = data_train[ target_name   ]
w_train = data_train[ "weight"      ]
X_valid = data_valid[ feature_names ]
y_valid = data_valid[ target_name   ]
w_valid = data_valid[ "weight"      ]



X_train_seq, y_train_seq = create_sequences(data_train, feature_names, target_name, n_steps)
X_valid_seq, y_valid_seq = create_sequences(data_valid, feature_names, target_name, n_steps)



# Parameters
sequence_length = 5
batch_size = 2000
lr = 0.01
weight_decay = 5e-4
# Input shape should be [n_steps, number of features]
input_shape = (n_steps, len(feature_names))


# Create model
model = create_lstm_model(input_shape=input_shape, lr=lr, weight_decay=weight_decay)
model.summary()



# Callbacks
ca = [
    tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=50, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/models/LSTM_Models/tf_LSTM_model_14_{epoch:02d}.keras',
        monitor='val_loss',
        save_best_only=False,  # Set to False to save every epoch
        save_freq='epoch'  # Save at the end of each epoch
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mse',
        factor=0.1,
        patience=20,
        verbose=1,
        min_lr=1e-6
    )
]


# Train the LSTM model with data shuffling
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_valid_seq, y_valid_seq),
    epochs=10,   # Choose your desired number of epochs
    batch_size=batch_size,
    shuffle=True  # Shuffle training data at start of each epoch
)








# This should be a NumPy array or a Tensor with shape (num_samples, num_features)
X_new = X_valid.to_numpy()  # Replace with actual data

# Make predictions
predictions = model.predict(X_new)
r2_metric = tf.keras.metrics.R2Score(class_aggregation='uniform_average')

if not isinstance(y_valid, np.ndarray):
    y_valid = y_valid.to_numpy()  # Convert to numpy array if it is a DataFrame

r2_metric.update_state(y_true=y_valid, y_pred=predictions)
r2_score_value = r2_metric.result().numpy()

print(f"R² Score on validation data: {r2_score_value}")