import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_random_seeds(seed=42):
    # Set the random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Before creating and training your model, call the function
set_random_seeds(42)


def load_data(path, start_dt, end_dt):
    data = pl.scan_parquet(path).select(
        pl.all(),).filter(
        pl.col("date_id").gt(start_dt),
        pl.col("date_id").le(end_dt),
    ).fill_null(0).fill_null(0)

    data = data.collect().to_pandas()

    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data

def making_time_id_features(X_test):
    X_test['sin_time_id']=np.sin(2*np.pi*X_test['time_id']/967)
    X_test['cos_time_id']=np.cos(2*np.pi*X_test['time_id']/967)
    X_test['sin_time_id_halfday']=np.sin(2*np.pi*X_test['time_id']/483)
    X_test['cos_time_id_halfday']=np.cos(2*np.pi*X_test['time_id']/483)
    return X_test

is_linux = False
if is_linux:
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    training_resp_lag_path = "/home/zt/pyProjects/JaneSt/Team/data/CustomData_2_RespLags_onTime/trainData"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/1_HS_Plan/Torch-NN-models"
else:
    training_resp_lag_path = "E:\Python_Projects\JS2024\GITHUB_C\data\CustomData_2_RespLags_onTime\\trainData"
    model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\9-Weired-thoughts\9-2-Model-SymbSpec\models"
    model_saving_name = "model_1.kears"
    
    

feature_names = ["symbol_id"] + [f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
time_id_features = ["sin_time_id", "cos_time_id", "sin_time_id_halfday", "cos_time_id_halfday"]
target_name = "responder_6"

X_train = load_data(training_resp_lag_path, start_dt=1450, end_dt=1500)
X_train = making_time_id_features(X_train)
y_train = X_train[target_name]
w_train = X_train["weight"]
X_train = X_train[feature_names+time_id_features]

X_valid = load_data(training_resp_lag_path, start_dt=1650, end_dt=1690)
X_valid = making_time_id_features(X_valid)
y_valid = X_valid[target_name]
w_valid = X_valid["weight"]
X_valid = X_valid[feature_names+time_id_features]



dimension = X_train.shape[1]


# Define your feature count — make sure you include `symbol_id` in this count
num_features = len(feature_names)  # Including 'symbol_id' in the feature count

# Input layer adjustment for non-sequential data: treat each row/sample as a separate input
input_layer = Input(shape=(num_features,), name='input_layer')

# Assume `symbol_id` is the first feature/column — adjust as needed based on your feature order
symbol_id = input_layer[:, 0]  # Directly using the first feature if it's an integer

# Process all features except the `symbol_id`
feature_input = input_layer[:, 1:]  # Exclude the `symbol_id` for shared processing
x = Dense(512, activation='swish')(feature_input)
x = Dense(512, activation='swish')(x)
x = Dense(256, activation='swish')(x)

# Process each symbol separately and collect outputs as before
symbol_outputs = []

for i in range(39):  # Adjust according to the number of possible symbols
    target_symbol_input = Lambda(lambda inputs: tf.where(
        tf.expand_dims(tf.equal(inputs[1], i), axis=-1),  # Expand dimensions to match `x`'s shape
        inputs[0],
        tf.zeros_like(inputs[0])
    ))([x, symbol_id])

    target_symbol_out = Dense(units=32, activation='swish', name=f"symbol_{i}")(target_symbol_input)
    symbol_outputs.append(target_symbol_out)

# Combine all symbol outputs
symbol_out_combined = tf.stack(symbol_outputs, axis=-1)  # Shape [N, 32, 39]
symbol_out_combined = tf.reduce_sum(symbol_out_combined, axis=-1)  # Shape [N, 32]
sigmoid_output = Dense(12, activation='swish')(symbol_out_combined)
# Final layer to make predictions
# output = Dense(1, activation='tanh')(symbol_out_combined)
sigmoid_output = Dense(1, activation='sigmoid')(symbol_out_combined)
scaled_output = Lambda(lambda x: x - 0.5)(sigmoid_output)

# Create the model
model = Model(inputs=input_layer, outputs=scaled_output)
# Model summary to verify the structure
model.summary()



model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=[tf.keras.metrics.R2Score(class_aggregation='uniform_average')])


lr = 0.01
weight_decay = 1e-6

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

model.predict(X_valid[:30])

