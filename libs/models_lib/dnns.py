"""

@author: Raffaele M Ghigliazza
"""

import os
import polars as pl
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import (layers, models, optimizers,
                              regularizers, callbacks)


def dnn_model(input_dim: int, lr: float,
              weight_decay: float,
              out_layer: str = 'tanh',
              simplified: bool = False,):

    regularizer = regularizers.l2(weight_decay)

    # Create a sequential model
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Add BatchNorm, ELU, Dropout, and Dense layers
    # BatchNormalization over the feature dimension (default for dense)

    if not simplified:
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Activation('swish'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(512, kernel_regularizer=regularizer))

        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Activation('swish'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(256, kernel_regularizer=regularizer))

    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(128, kernel_regularizer=regularizer))

    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Activation('swish'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, kernel_regularizer=regularizer))

    # Output layer
    model.add(layers.Dense(1, activation=out_layer))

    # Compile model with Mean Squared Error loss
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='mse',
                  metrics=[tf.keras.metrics.R2Score(
                      class_aggregation='uniform_average')])

    return model



# def create_model(input_dim, lr, weight_decay):
#     # Create a sequential model
#     model = models.Sequential()
#
#     # Assuming input_dim represents features for 1D data, reshape it for Conv1D
#     model.add(layers.Input(shape=(input_dim, 1)))
#     # Add a Conv1D layer
#     model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='swish'))
#     # Optionally, add another layer such as MaxPooling1D
#     model.add(layers.MaxPooling1D(pool_size=2))
#     # Flatten before the Dense layers
#     model.add(layers.Flatten())
#
#     # Add BatchNorm, SiLU (Swish in TensorFlow), Dropout, and Dense (Linear) layers
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation('swish'))
#     model.add(layers.Dropout(0.1))
#     model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
#
#     # Add subsequent hidden layers in a flattened structure
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation('swish'))
#     model.add(layers.Dropout(0.1))
#     model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(weight_decay)))
#
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation('swish'))
#     model.add(layers.Dropout(0.1))
#     model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(weight_decay)))
#
#     # Output layer
#     model.add(layers.Dense(1, activation='tanh'))
#
#     # Compile model with Mean Squared Error loss
#     model.compile(optimizer=optimizers.Adam(learning_rate=lr),
#                   loss='mse',
#                   metrics=[tf.keras.metrics.R2Score(class_aggregation='uniform_average')])
#
#     return model
