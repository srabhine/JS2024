
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
