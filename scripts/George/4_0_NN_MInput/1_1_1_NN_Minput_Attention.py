import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Permute, Dense, Add, Activation, Lambda, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1],),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        x_score = K.tanh(K.dot(x, self.W) + self.b)
        attention_weights = K.softmax(x_score, axis=1)
        output = x * attention_weights
        return output


def create_model(input_dimensions, lr):
    input_group_1 = Input(shape=(input_dimensions[0],), name='input_group_1')
    input_group_2 = Input(shape=(input_dimensions[1],), name='input_group_2')
    input_group_3 = Input(shape=(input_dimensions[2],), name='input_group_3')

    # Define sub-network for group 1 with attention
    x1 = Dense(128, activation='swish')(input_group_1)
    x1 = AttentionLayer()(x1)
    x1 = Dense(64, activation='swish')(x1)

    # Define sub-network for group 2 with attention
    x2 = Dense(128, activation='swish')(input_group_2)
    x2 = AttentionLayer()(x2)
    x2 = Dense(64, activation='swish')(x2)

    # Define sub-network for group 3 with attention
    x3 = Dense(128, activation='swish')(input_group_3)
    x3 = AttentionLayer()(x3)
    x3 = Dense(64, activation='swish')(x3)

    # Combine outputs from each sub-network
    combined = Concatenate()([x1, x2, x3])
    combined = Dense(32, activation='swish')(combined)

    # Define the output layer
    output = Dense(1, activation='tanh', name='output')(combined)

    # Create the model
    model = Model(inputs=[input_group_1, input_group_2, input_group_3], outputs=output)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=[tf.keras.metrics.R2Score()])

    # Print summary of the model
    model.summary()

    return model


def load_data(train_path, valid_path, start_id, end_id):
    # df = pl.scan_parquet(f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/FOLD3").collect().to_pandas()
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


train_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/NN_Models/data/Data_date_id_Partition/training.parquet"
valid_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/NN_Models/data/Data_date_id_Partition/validation.parquet"
model_saving_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/NN_Models/models/model_1_multi_group"
model_saving_name = "model_1_Minput_Attention_{epoch:02d}.keras"

feature_names = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'
start_id = 1000
end_id = 1646
data_train, data_valid = load_data(train_path, valid_path, start_id, end_id)

X_train = data_train[ feature_names ]
y_train = data_train[ label_name    ]
w_train = data_train[ "weight"      ]
X_valid = data_valid[ feature_names ]
y_valid = data_valid[ label_name    ]
w_valid = data_valid[ "weight"      ]


feature_1 = [f"feature_{i:02d}" for i in range(0,  20)]
feature_2 = [f"feature_{i:02d}" for i in range(20, 79)]
feature_3 = [f"responder_{idx}_lag_1" for idx in range(9)]

X_train_1 = X_train[ feature_1 ].to_numpy()
X_train_2 = X_train[ feature_2 ].to_numpy()
X_train_3 = X_train[ feature_3 ].to_numpy()

X_valid_1 = X_valid[ feature_1 ].to_numpy()
X_valid_2 = X_valid[ feature_2 ].to_numpy()
X_valid_3 = X_valid[ feature_3 ].to_numpy()


lr = 0.01
input_dimensions = [X_train_1.shape[1], X_train_2.shape[1], X_train_3.shape[1]]
model = create_model(input_dimensions, lr)


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
    x=[X_train_1, X_train_2, X_train_3],  # Input features for training
    y=y_train,                          # Target labels for training
    sample_weight=w_train,              # Sample weights for training
    validation_data=([X_valid_1, X_valid_2, X_valid_3], y_valid, w_valid),  # Validation data
    batch_size=1028,                      # Batch size
    epochs=10,                        # Number of epochs
    callbacks=ca,                # Callbacks list, if any
    verbose=1,                           # Verbose output during training
    shuffle=True
)


##


# Assume X_new is your new data you want to make predictions on
# This should be a NumPy array or a Tensor with shape (num_samples, num_features)
# X_new = X_valid.to_numpy()  # Replace with actual data

# Make predictions
predictions = model.predict([X_valid_1, X_valid_2, X_valid_3],)
r2_metric = tf.keras.metrics.R2Score(class_aggregation='uniform_average')

r2_metric.update_state(y_true=y_valid, y_pred=predictions)
r2_score_value = r2_metric.result().numpy()

print(f"R² Score on validation data: {r2_score_value}")




# ReLoad the model

@tf.keras.utils.register_keras_serializable(package='Custom', name='AttentionLayer')
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        x_score = K.tanh(K.dot(x, self.W) + self.b)
        attention_weights = K.softmax(x_score, axis=1)
        output = x * attention_weights
        return output

    def get_config(self):
        base_config = super(AttentionLayer, self).get_config()
        return base_config


model = tf.keras.models.load_model(
    '/home/zt/pyProjects/Optiver/JaneStreetMktPred/NN_Models/models/model_1_multi_group/model_1_Minput_Attention_35.keras',
    custom_objects={'AttentionLayer': AttentionLayer}
)




predictions = model.predict([X_valid_1, X_valid_2, X_valid_3],)
r2_metric = tf.keras.metrics.R2Score(class_aggregation='uniform_average')
r2_metric.update_state(y_true=y_valid, y_pred=predictions)
r2_score_value = r2_metric.result().numpy()
print(f"R² Score on validation data: {r2_score_value}")

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
    x=[X_train_1, X_train_2, X_train_3],  # Input features for training
    y=y_train,  # Target labels for training
    sample_weight=w_train,  # Sample weights for training
    validation_data=([X_valid_1, X_valid_2, X_valid_3], y_valid, w_valid),  # Validation data
    batch_size=1028,  # Batch size
    epochs=100,  # Total epochs including previous training (e.g., 20 to end after additional 10 epochs)
    initial_epoch=10,  # Start from epoch 10
    callbacks=ca,  # Callbacks list, if any
    verbose=1,  # Verbose output during training
    shuffle=True  # To shuffle data
)