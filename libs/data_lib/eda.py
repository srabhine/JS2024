import pandas as pd
from tqdm import tqdm


def create_symbol_dataframes(df: pd.DataFrame):
    """
    Analyse tous les symbols et leurs données chronologiques
    Create a dictionary of data frames per symbol

    Args:
        df: DataFrame contenant les données; original data frame e.g. part-0.parquet
    Returns:
        Dict avec les données groupées par symbol
    """
    # Obtenir la liste unique des symbol_ids
    unique_symbols = sorted(df['symbol_id'].unique())
    print(f"Nombre total de symbols: {len(unique_symbols)}")

    # Dictionnaire pour stocker les résultats
    symbol_data = {}

    # Pour chaque symbol_id
    for symbol in tqdm(unique_symbols):
        # Filtrer les données pour ce symbol
        symbol_df = (df[df['symbol_id'] == symbol]
                     .sort_values('date_id')
                     .copy())

        # Stocker dans le dictionnaire
        symbol_data[symbol] = symbol_df

    return symbol_data

