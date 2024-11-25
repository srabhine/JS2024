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
