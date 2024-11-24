"""

@authors: George, Raffaele
"""
# import sys
# sys.path.append('/home/zt/pyProjects/JaneSt/Team/libs')
import os
import polars as pl
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from data_lib.core_tf import prepare_dataset, defined_callbacks
from data_lib.datasets import get_data_by_symbol, \
    get_features_classification
from data_lib.random_gen import set_seed
from data_lib.variables import FEATS_TIME_LAG, RESP_DAY_LAG, FEATS, \
    TARGET, FEATS_TOP_50
from io_lib.paths import MODELS_DIR
from models_lib.dnns import dnn_model

case = 'all'
# case = 'feats'
# case = 'feats_time_lag'
# case = 'resp_day_lag'
# feature_names = FEATS + FEATS_TIME_LAG + RESP_DAY_LAG
# feature_names = FEATS


sym = 1
out_layer = 'tanh'
# out_layer = 'linear'


cases = ['all', 'feats', 'feats_time_lag',
         'resp_day_lag', 'top_50',
         'cleanup', 'normalize', 'transform']
r2 = {}
for case in cases:
    is_transform = False
    feat_types_dic = None
    if case == 'feats':
        feature_names = FEATS
    elif case == 'feats_time_lag':
        feature_names = FEATS_TIME_LAG
    elif case == 'resp_day_lag':
        feature_names = RESP_DAY_LAG
    elif case == 'top_50':
        feature_names = FEATS_TOP_50
    elif case == 'all':
        feature_names = FEATS + FEATS_TIME_LAG + RESP_DAY_LAG
    elif case == 'normalize':
        is_transform = True
        feature_names = FEATS
    elif case == 'cleanup':
        is_transform = True
        feat_types_dic = 'cleanup'
        feature_names = FEATS
    elif case == 'transform':
        is_transform = True
        feat_types_dic = get_features_classification()
        feature_names = FEATS
    else:
        raise ValueError('Invalid case')

    (df_sym, vld_sym, X_train, y_train, w_train,
     X_valid, y_valid, w_valid) = \
        get_data_by_symbol(feature_names, sym=sym,
                           is_transform=is_transform,
                           feat_types_dic=feat_types_dic)

    # Set seed
    set_seed(0)

    train_dataset = prepare_dataset(df_sym, w_train, feature_names,
                                    TARGET, batch_size=8129)
    valid_dataset = prepare_dataset(vld_sym, w_valid, feature_names,
                                    TARGET, batch_size=8129)

    lr = 0.01
    weight_decay = 5e-4
    # weight_decay = 0

    input_dim = df_sym[feature_names].shape[1]

    model = dnn_model(input_dim=input_dim, lr=lr,
                      weight_decay=weight_decay,
                      out_layer=out_layer,
                      simplified=True)
    model.summary()

    suffix = (f'/dnn_v10_{out_layer}_transf_{is_transform}'
              f'_{case}.keras')
    path = str(MODELS_DIR) + suffix

    ca = defined_callbacks(path)
    model.fit(
        train_dataset.map(
            lambda x, y, w: (x, y, {'sample_weight': w})),
        epochs=70,
        validation_data=valid_dataset.map(
            lambda x, y, w: (x, y, {'sample_weight': w})),
        callbacks=ca
    )
    model.save(path)

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

    print(f"R² Score on validation data: {r2_score_value:1.6f}")

    r2[case] = r2_score_value


print(r2)
print(pd.Series(r2))
