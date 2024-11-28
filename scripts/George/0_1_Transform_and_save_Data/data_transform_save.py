
from one_big_lib import transform_data
from data_lib.datasets import get_features_classification
import polars as pl
from features_lib.core import transform_features
import pickle



# feature_names = FEATS
feat_types_dic = get_features_classification()
local_path = "/home/zt/pyProjects/JaneSt/Team/data/transformed_data"
# local_path ="E:\Python_Projects\JS2024\GITHUB_C\data\\transformed_data2"

for i in range(1,10):
    path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}"
        # f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet/partition_id={i}"

    
        
    data = pl.scan_parquet(path).collect().to_pandas()
    feat_types_dic['responder_6'] = 'none'
    data_transf, params = transform_data(data, transformation=feat_types_dic)
    
    data_transf = pl.DataFrame(data_transf)

    data_transf.write_parquet(
        # f"E:\Python_Projects\JS2024\GITHUB_C\data\\transformed_data2\\train_parquet_{i}.parquet",
        f"{local_path}/train_parquet_{i}.parquet"
    )

    
    with open(f"{local_path}/params_{i}.pkl", 'wb') as file:
        pickle.dump(params, file)


