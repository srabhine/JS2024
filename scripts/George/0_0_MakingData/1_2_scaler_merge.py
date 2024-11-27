import pandas as pd
import numpy as np

def pool_scalers(path, start, end):
	scaler_dataframes = []
	for i in range(start, end+1):  # From 1 to 9
		df = pd.read_csv(path.format(i), index_col=0)
		df = df.reset_index().rename(columns={'index': 'feature', '0': 'scaler'})
		# Add a suffix to identify the DataFrame
		df = df.add_suffix(f'_{i}')
		df.rename(columns={f'feature_{i}': 'feature', f'symbol_id_{i}': 'symbol_id'}, inplace=True)
		scaler_dataframes.append(df)
	
	# Merge all DataFrames on 'feature' and 'symbol_id'
	merged_data = scaler_dataframes[0]
	for df in scaler_dataframes[1:]:
		merged_data = pd.merge(merged_data, df, on=['feature', 'symbol_id'], how='outer')
	
	# Calculate the mean of all 'scaler_mu' columns
	merged_data = merged_data.sort_values(by=['feature', 'symbol_id'])
	return merged_data



file_path_mu = "E:/Python_Projects/JS2024/GITHUB_C/data/transformed_data/scalers_mu{}.csv"
merged_data = pool_scalers(path=file_path_mu, start=1, end=9)
scaler_columns = [col for col in merged_data.columns if 'scaler' in col]
merged_data['0'] = merged_data[scaler_columns].mean(axis=1)
scaler_mu = merged_data[['feature', 'symbol_id', '0']]
scaler_mu.set_index(['feature'], inplace=True)
scaler_mu.index.name = None
scaler_mu.to_csv('E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_0_MakingData\made_scalers\scalers_mu.csv')



file_path_sg = "E:/Python_Projects/JS2024/GITHUB_C/data/transformed_data/scalers_sg{}.csv"
merged_data = pool_scalers(path=file_path_sg, start=1, end=9)
scaler_columns = [col for col in merged_data.columns if 'scaler' in col]
merged_data['0'] = np.sqrt((merged_data[scaler_columns]**2).mean(axis=1))
scaler_sg = merged_data[['feature', 'symbol_id', '0']]
scaler_sg.set_index(['feature'], inplace=True)
scaler_sg.index.name = None
scaler_sg.to_csv('E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_0_MakingData\made_scalers\scalers_sg.csv')


# test1 = pd.read_csv('E:\Python_Projects\JS2024\GITHUB_C\data\\transformed_data\scaler_mu1.csv', index_col=0)
# test2 = pd.read_csv('E:\Python_Projects\JS2024\GITHUB_C\data\\transformed_data\scaler_mu2.csv', index_col=0)