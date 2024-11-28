import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import polars as pl

SYMBOLS = list(range(39))
RESPONDERS = list(range(9))
IX_IDS_BY_SYM = ['symbol_id', 'date_id', 'time_id']
FEATS = [f"feature_{i:02d}" for i in range(79)]
TARGET = 'responder_6'

path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet/partition_id=5"
data = pl.scan_parquet(path).collect().to_pandas()



