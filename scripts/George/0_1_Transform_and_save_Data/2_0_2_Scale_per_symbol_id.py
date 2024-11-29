import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import polars as pl
from typing import Optional, List, Union, Dict, Any, Tuple
import os
import pickle
import numpy as np



SYMBOLS = list(range(39))
RESPONDERS = list(range(9))
IX_IDS_BY_SYM = ['symbol_id', 'date_id', 'time_id']
FEATS = [f"feature_{i:02d}" for i in range(79)]
TARGET = 'responder_6'


#
# path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet/partition_id=5"
# data = pl.scan_parquet(path).collect().to_pandas()

def load_data():
    path_win = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    path_linux = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    start_dt = 1350
    end_dt = 1400
    data = pl.scan_parquet(path_win
                           ).select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
        pl.all(),
    ).filter(
        pl.col("date_id").gt(start_dt),
        pl.col("date_id").le(end_dt),
    )

    data = data.collect().to_pandas()

    data.replace([np.inf, -np.inf], 0, inplace=True)
    data = data.fillna(0)
    return data




features_to_scale = ['feature_01', 'feature_04','feature_18','feature_19','feature_33','feature_36','feature_39','feature_40',
                     'feature_41','feature_42','feature_43', 'feature_44','feature_45','feature_46','feature_50','feature_51',
                     'feature_52','feature_53','feature_54','feature_55','feature_56','feature_57','feature_63','feature_64',
                     'feature_78']

data = load_data()

# def normalize_features(data, features_to_scale):
# 	# List all features for consistency
# 	all_features = [f"feature_{str(x).zfill(2)}" for x in range(80)]
#
# 	# Initialize DataFrames for storing means and stds for each feature per symbol_id
# 	scaler_mean_df = pd.DataFrame(index=pd.Index(data['symbol_id'].unique(), name='symbol_id'),
# 	                              columns=all_features).fillna(0.0)
# 	scaler_std_df = pd.DataFrame(index=pd.Index(data['symbol_id'].unique(), name='symbol_id'),
# 	                             columns=all_features).fillna(1.0)
#
# 	# Group by symbol_id
# 	grouped_data = data.groupby('symbol_id')
#
# 	# Normalize features per symbol_id
# 	for symbol_id, group in grouped_data:
# 		# Calculate mean and std for each feature in features_to_scale
# 		means = group[features_to_scale].mean()
# 		stds = group[features_to_scale].std()
#
# 		# Update the corresponding rows in the DataFrames
# 		scaler_mean_df.loc[symbol_id, features_to_scale] = means
# 		scaler_std_df.loc[symbol_id, features_to_scale] = stds
#
# 	return scaler_mean_df, scaler_std_df


def normalize_features(data, features_to_scale):
	# List all features for consistency
	all_features = [f"feature_{str(x).zfill(2)}" for x in range(80)]
	
	# Construct symbol_id ranging from 0 to 50
	symbol_id_range = range(80)  # This creates a range from 0 to 50
	
	# Initialize DataFrames for storing means and stds for each feature per symbol_id
	scaler_mean_df = pd.DataFrame(index=pd.Index(symbol_id_range, name='symbol_id'), columns=all_features).fillna(0.0)
	scaler_std_df = pd.DataFrame(index=pd.Index(symbol_id_range, name='symbol_id'), columns=all_features).fillna(1.0)
	
	# Group by symbol_id
	grouped_data = data.groupby('symbol_id')
	
	# Normalize features per symbol_id
	for symbol_id, group in grouped_data:
		if symbol_id in symbol_id_range:  # Only process symbol_ids within the desired range
			# Calculate mean and std for each feature in features_to_scale
			means = group[features_to_scale].mean()
			stds = group[features_to_scale].std()
			
			# Update the corresponding rows in the DataFrames
			scaler_mean_df.loc[symbol_id, features_to_scale] = means
			scaler_std_df.loc[symbol_id, features_to_scale] = stds
	
	return scaler_mean_df, scaler_std_df

scaler_mean_df, scaler_std_df = normalize_features(data, features_to_scale)


def calculate_and_fill_mean_with_average(scaler_mean_df, features_to_scale):
    # Calculate the average mean from symbol_id 0 to 38
    average_mean = scaler_mean_df.loc[0:38, features_to_scale].mean()

    # Fill rows 39 to 50 with the average mean
    for symbol_id in range(39, 80):
        scaler_mean_df.loc[symbol_id, features_to_scale] = average_mean

    return scaler_mean_df




def calculate_and_fill_std_with_average(scaler_std_df, features_to_scale):
	# Calculate the pooled standard deviation for symbol_id 0 to 38
	pooled_std_series = pd.Series(index=features_to_scale, dtype=np.float64)
	variances = scaler_std_df.loc[0:38, features_to_scale] ** 2
	mean_variance = variances.mean()
	pooled_std_series[features_to_scale] = np.sqrt(mean_variance)
	
	# Fill symbol_id 39 to 79 with the pooled standard deviation
	for symbol_id in range(39, 80):
		scaler_std_df.loc[symbol_id, features_to_scale] = pooled_std_series[features_to_scale]
	
	return scaler_std_df
	

	
	
	return pooled_std_series

scaler_mean_df = calculate_and_fill_mean_with_average(scaler_mean_df, features_to_scale)
scaler_std_df = calculate_and_fill_std_with_average(scaler_std_df, features_to_scale)

merged_scalers_df = pd.merge(scaler_mean_df, scaler_std_df, on='symbol_id', how = 'outer', suffixes=('_mean', '_std'))



pickle_file = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\\merged_scalers_df.pkl"
with open(pickle_file, 'wb') as f:
    pickle.dump(merged_scalers_df, f)
    
