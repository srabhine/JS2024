
from data_lib.datasets import get_data_by_symbol, \
    get_features_classification
from data_lib.random_gen import set_seed
from data_lib.variables import FEATS_TIME_LAG, RESP_DAY_LAG, FEATS, \
    TARGET, FEATS_TOP_50, SYMBOLS
import polars as pl
from features_lib.core import transform_features


# case = 'all'
# case = 'feats'
# case = 'feats_time_lag'
# case = 'resp_day_lag'
# feature_names = FEATS + FEATS_TIME_LAG + RESP_DAY_LAG
# feature_names = FEATS
#
#
# # sym = 2
# sym = SYMBOLS
# # sym = [3, 4]
# out_layer = 'tanh'
# # out_layer = 'linear'


# cases = ['all', 'feats', 'feats_time_lag',
#          'resp_day_lag', 'top_50',
#          'cleanup', 'normalize', 'transform']
# cases = ['cleanup']
# cases = ['cleanup', 'normalize']
# cases = ['normalize', 'transform']


# feature_names = FEATS
feat_types_dic = get_features_classification()
# path = "E:\Python_Projects\JS2024\GITHUB_C\data\\non_lag_data\\training_parquet" \
#        "\\date_id=1501\\*.parquet"

for i in range(1,10):
    path =f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}"
    df = pl.scan_parquet(path).collect().to_pandas()
    
    df, scalers_mu, scalers_sg = transform_features(df, feat_types_dic)
    df = pl.DataFrame(df)

    df.write_parquet(
        f"/home/zt/pyProjects/JaneSt/Team/data/transformed_data/train_parquet_{i}.parquet",
    )
    scalers_mu.to_csv(f"/home/zt/pyProjects/JaneSt/Team/data/transformed_data/scaler_mu{i}")
    scalers_sg.to_csv(f"/home/zt/pyProjects/JaneSt/Team/data/transformed_data/scalers_sg{i}")

