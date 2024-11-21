"""

@author: George
"""

import numpy as np
import tensorflow as tf


def defined_callbacks(path: str):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_r2_score',
                                         patience=15, mode='max'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=path,
            monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_r2_score',  # Metric to be monitored
            factor=0.1,  # Factor by which the learning rate will be reduced
            patience=8,  # Number of epochs with no improvement after which learning rate will be reduced
            verbose=1,  # Verbosity mode
            min_lr=1e-6  # Lower bound on the learning rate
        )
    ]


def get_generator_v3(dataframe, weights, feature_names,
                     label_name, shuffle=True, batch_size=8192):
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


def prepare_dataset(dataframe, weights, feature_names,
                    label_name, batch_size=8192, shuffle=True):
    num_features = len(feature_names)

    output_signature = (
        tf.TensorSpec(shape=(None, num_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        get_generator_v3(dataframe, weights, feature_names,
                         label_name, shuffle, batch_size),
        output_signature=output_signature
    )

    return dataset
