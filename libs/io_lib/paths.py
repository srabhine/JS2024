
# Google drive:
# https://drive.google.com/file/d/1a3NX9wsXCBKscJO43WUEPHuxGbIP40Dh
# https://drive.google.com/drive/folders/135BWdBY-d0fp7jYqKxoaToLVS5g3rGV5
# https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/543567


# Paths
import os
from pathlib import Path
from datetime import date
import datetime
# from pandas.tseries.offsets import BDay

ROOT = Path(__file__).parent.parent.parent
print(ROOT)

LIB_DIR = ROOT / 'libs'
DATA_DIR = ROOT / 'data'
FIGS_DIR = ROOT / 'figures'


TRAIN_DIR = DATA_DIR / 'train_parquet'
LAGS_FEAT_DIR = DATA_DIR / 'lags_features'
LAGS_FEATURES_TRAINING = LAGS_FEAT_DIR / 'training_parquet'
LAGS_FEATURES_VALIDATION = LAGS_FEAT_DIR /'validation_parquet'
