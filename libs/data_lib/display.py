from typing import Dict

import pandas as pd


def display_symbol_timeline(symbol_data: Dict[str, pd.DataFrame],
                            n_symbols: int = 5,
                            n_dates: int = 5):
    """
    Affiche les premières dates pour plusieurs symbols
    """
    for symbol in list(symbol_data.keys())[:n_symbols]:
        print(f"\nSymbol ID: {symbol}")
        print("-" * 50)
        print(symbol_data[symbol]
              .sort_values('date_id')
              [['date_id', 'symbol_id', 'weight'] +
               [col for col in symbol_data[symbol].columns if col.startswith('feature')]]
              .head(n_dates))


def get_symbol_statistics(symbol_data: Dict[str, pd.DataFrame]):
    """
    Calcule les statistiques pour chaque symbol
    """
    stats = []

    for symbol, data in symbol_data.items():
        stats.append({
            'symbol_id': symbol,
            'n_dates': len(data),
            'mean_weight': data['weight'].mean(),
            'min_date': data['date_id'].min(),
            'max_date': data['date_id'].max(),
            'n_features': len([col for col in data.columns if
                               col.startswith('feature')])
        })

    return pd.DataFrame(stats)

