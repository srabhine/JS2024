
from one_big_lib import transform_data_2
from data_lib.datasets import get_features_classification
import polars as pl
from features_lib.core import transform_features




# feature_names = FEATS
feat_types_dic = get_features_classification()


for i in range(1,10):
    path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet/partition_id={i}"
        # f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}"
    
        
    df = pl.scan_parquet(path).collect().to_pandas()
    
    df, scalers_mu, scalers_sg = transform_data_2(df, transformation=feat_types_dic)
    df = pl.DataFrame(df)

    df.write_parquet(
        f"E:\Python_Projects\JS2024\GITHUB_C\data\\transformed_data\\train_parquet_{i}.parquet",
    )
    scalers_mu.to_csv(f"E:\Python_Projects\JS2024\GITHUB_C\data\\transformed_data\\scalers_mu{i}.csv")
    scalers_sg.to_csv(f"E:\Python_Projects\JS2024\GITHUB_C\data\\transformed_data\\scalers_sg{i}.csv")

