import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path

# NEED APPROX 30 GB OF RAM TO RUN THIS SCRIPT bei Nicks Test

# Base path where partitioned symbol data is stored
base_path = r'E:\coding projects\2024\jane street 2024\jane-street-real-time-market-data-forecasting\train sorted by symbol id'

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory in the same folder as this script
output_dir = os.path.join(script_dir, 'merged_symbols_dataset')
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store DataFrames for each symbol
symbol_data = {}

# Iterate through all partition directories
for partition_id in range(10):
    partition_path = os.path.join(base_path, f'partition_id={partition_id}')
    
    # Get all parquet files in this partition
    symbol_files = list(Path(partition_path).glob('symbol_*.parquet'))
    
    print(f"\nProcessing partition {partition_id}")
    for symbol_file in tqdm(symbol_files, desc=f"Reading symbols from partition {partition_id}"):
        # Extract symbol ID from filename
        symbol_id = int(symbol_file.stem.split('_')[1])
        
        # Read the parquet file
        df = pd.read_parquet(symbol_file)
        
        if symbol_id in symbol_data:
            # Append to existing data
            symbol_data[symbol_id] = pd.concat([symbol_data[symbol_id], df], ignore_index=True)
        else:
            # Create new entry
            symbol_data[symbol_id] = df

print("\nMerging and cleaning symbol data...")
for symbol_id in tqdm(symbol_data.keys(), desc="Processing symbols"):
    # Sort by date_id and time_id
    symbol_data[symbol_id] = (symbol_data[symbol_id]
                             .sort_values(['date_id', 'time_id'])
                             .drop_duplicates()  # Remove any potential duplicates
                             .reset_index(drop=True))

print("\nSaving merged symbol datasets...")
for symbol_id, df in tqdm(symbol_data.items(), desc="Saving merged symbols"):
    output_path = os.path.join(output_dir, f'symbol_{symbol_id}_merged.parquet')
    df.to_parquet(output_path)

# Print summary statistics
print("\nSummary:")
for symbol_id, df in symbol_data.items():
    print(f"Symbol {symbol_id}: {len(df)} rows, Date range: {df['date_id'].min()} to {df['date_id'].max()}")

print(f"\nMerged datasets have been saved to: {output_dir}")

