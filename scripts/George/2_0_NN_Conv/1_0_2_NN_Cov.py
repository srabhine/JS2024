import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks



# Prepare datasets

def get_generator_v3(dataframe, weights, feature_names, label_name, shuffle=True, batch_size=8192):
    def generator():
        indices = np.arange(len(dataframe))
        if shuffle:
            np.random.shuffle(indices)

        num_batches = len(indices) // batch_size + (1 if len(indices) % batch_size > 0 else 0)

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(dataframe))
            current_indices = indices[start_index:end_index]

            features = dataframe.iloc[current_indices][feature_names].values
            labels = dataframe.iloc[current_indices][label_name].values
            if weights is not None:
                weights_batch = weights.iloc[current_indices].values
            else:
                weights_batch = np.ones(len(labels), dtype=np.float32)

            yield features, labels.reshape(-1, 1), weights_batch

    return generator


def prepare_dataset(dataframe, weights, feature_names, label_name, batch_size=8192, shuffle=True):
    num_features = len(feature_names)

    output_signature = (
        tf.TensorSpec(shape=(None, num_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        get_generator_v3(dataframe, weights, feature_names, label_name, shuffle, batch_size),
        output_signature=output_signature
    )

    return dataset

def create_model(input_dim, lr, weight_decay):
    # Create a sequential model
    model = models.Sequential()

    # Assuming input_dim represents features for 1D data, reshape it for Conv1D
    model.add(layers.Input(shape=(input_dim, 1)))
    # Add a Conv1D layer
    model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='swish'))
    # Optionally, add another layer such as MaxPooling1D
    model.add(layers.MaxPooling1D(pool_size=2))
    # Flatten before the Dense layers
    model.add(layers.Flatten())

    # Add BatchNorm, SiLU (Swish in TensorFlow), Dropout, and Dense (Linear) layers
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))

    # Add subsequent hidden layers in a flattened structure
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(layers.BatchNormalization())
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(weight_decay)))

    # Output layer
    model.add(layers.Dense(1, activation='tanh'))

    # Compile model with Mean Squared Error loss
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=[tf.keras.metrics.R2Score(class_aggregation='uniform_average')])

    return model








# Load data
# df = pl.scan_parquet(f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/FOLD3").collect().to_pandas()

def load_data(path, start_dt, end_dt):
    data = pl.scan_parquet(path).select(
        pl.all(),).filter(
        pl.col("date_id").gt(start_dt),
        pl.col("date_id").le(end_dt),
    ).fill_null(0).fill_null(0)

    data = data.collect().to_pandas()

    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data

training_resp_lag_path = "/home/zt/pyProjects/JaneSt/Team/data/CustomData_1_RespLags/training"
feature_names = ["symbol_id"] + [f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
time_id_features = ["sin_time_id", "cos_time_id", "sin_time_id_halfday", "cos_time_id_halfday"]
target_name = "responder_6"





X_train = load_data(training_resp_lag_path, start_dt=500, end_dt=1600)
y_train = X_train[target_name]
w_train = X_train["weight"]

X_valid = load_data(training_resp_lag_path, start_dt=500, end_dt=1690)
y_valid = X_valid[target_name]
w_valid = X_valid["weight"]



train_dataset = prepare_dataset(X_train, w_train, feature_names, target_name, batch_size=10200)
valid_dataset = prepare_dataset(X_valid, w_valid, feature_names, target_name, batch_size=10200)




lr = 0.01
weight_decay = 5e-4

input_dim = X_valid[feature_names].shape[1]
model = create_model(input_dim=input_dim, lr = lr, weight_decay=weight_decay)
model.summary()





ca = [
    tf.keras.callbacks.EarlyStopping(monitor='val_r2_score', patience=25, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/models/tf_nn_ELU_7.keras',
        monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Metric to be monitored
        factor=0.1,  # Factor by which the learning rate will be reduced
        patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1,  # Verbosity mode
        min_lr=1e-6  # Lower bound on the learning rate
    )

]

model.fit(
    train_dataset.map(lambda x, y, w: (x, y, {'sample_weight': w})),
    epochs=2000,
    validation_data=valid_dataset.map(lambda x, y, w: (x, y, {'sample_weight': w})),
    callbacks=ca
)



##


# Assume X_new is your new data you want to make predictions on
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

