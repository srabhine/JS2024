# Cell 0 - Imports
import pandas as pd
import polars as pl
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Cell 1 - Analysis Functions
def analyze_all_symbols(df):
    unique_symbols = sorted(df['symbol_id'].unique())
    print(f"Nombre total de symbols: {len(unique_symbols)}")
    
    symbol_data = {}
    for symbol in tqdm(unique_symbols):
        symbol_df = (df[df['symbol_id'] == symbol]
                    .sort_values('date_id')
                    .copy())
        symbol_data[symbol] = symbol_df
        
    return symbol_data

def display_symbol_timeline(symbol_data, n_symbols=5, n_dates=5):
    for symbol in list(symbol_data.keys())[:n_symbols]:
        print(f"\nSymbol ID: {symbol}")
        print("-" * 50)
        print(symbol_data[symbol]
              .sort_values('date_id')
              [['date_id', 'symbol_id', 'weight'] + 
               [col for col in symbol_data[symbol].columns if col.startswith('feature')]]
              .head(n_dates))

def plot_feature_evolution(symbol_data, feature_name='feature_05', n_symbols=5):
    plt.figure(figsize=(15, 8))
    
    for symbol in list(symbol_data.keys())[:n_symbols]:
        data = symbol_data[symbol].sort_values('date_id')
        plt.plot(data['date_id'], data[feature_name], 
                label=f'Symbol {symbol}', alpha=0.7)
    
    plt.title(f'Évolution de {feature_name} par Symbol')
    plt.xlabel('Date ID')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    plt.close()

def get_symbol_statistics(symbol_data):
    stats = []
    for symbol, data in symbol_data.items():
        stats.append({
            'symbol_id': symbol,
            'n_dates': len(data),
            'mean_weight': data['weight'].mean(),
            'min_date': data['date_id'].min(),
            'max_date': data['date_id'].max(),
            'n_features': len([col for col in data.columns if col.startswith('feature')])
        })
    return pd.DataFrame(stats)

def run_complete_analysis(df, n_display_symbols=5, feature_to_plot='feature_05'):
    print("1. Groupement des données par symbol...")
    symbol_data = analyze_all_symbols(df)
    
    print("\n2. Aperçu des données chronologiques:")
    display_symbol_timeline(symbol_data, n_symbols=n_display_symbols)
    
    print("\n3. Statistiques par symbol:")
    stats_df = get_symbol_statistics(symbol_data)
    print(stats_df.head())
    
    print("\n4. Visualisation de l'évolution d'une feature:")
    plot_feature_evolution(symbol_data, feature_name=feature_to_plot, 
                         n_symbols=n_display_symbols)
    
    return symbol_data, stats_df

def create_symbol_datasets(df_symbols, base_path):
    os.makedirs(base_path, exist_ok=True)
    unique_symbols = sorted(df_symbols['symbol_id'].unique())
    for symbol in tqdm(unique_symbols, desc="Saving symbol datasets"):
        # Get data for this symbol and sort by date_id and time_id
        symbol_df = (df_symbols[df_symbols['symbol_id'] == symbol]
                    .sort_values(['date_id', 'time_id'])
                    .reset_index(drop=True))
        output_path = os.path.join(base_path, f"symbol_{symbol}.parquet")
        symbol_df.to_parquet(output_path)

# Cell 2 - Main Processing Function
def process_partition(partition_id):
    input_path = fr'E:\coding projects\2024\jane street 2024\jane-street-real-time-market-data-forecasting\train.parquet\partition_id={partition_id}\part-0.parquet'
    output_base_path = fr'E:\coding projects\2024\jane street 2024\jane-street-real-time-market-data-forecasting\train sorted by symbol id\partition_id={partition_id}'
    
    print(f"\nProcessing partition {partition_id}")
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_base_path}")
    
    df = pd.read_parquet(input_path)
    symbol_data, stats = run_complete_analysis(df)
    df_symbols = pd.concat(symbol_data.values(), ignore_index=True).sort_values(['symbol_id', 'date_id']).reset_index(drop=True)
    create_symbol_datasets(df_symbols, output_base_path)

# Cell 3 - Main Execution
if __name__ == "__main__":
    for partition_id in range(10):
        process_partition(partition_id)