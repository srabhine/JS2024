from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt

from libs.io_lib.paths import FIGS_DIR


def plot_feature_evolution(symbol_data: Dict[str, pd.DataFrame],
                           feature_name: str ='feature_05',
                           n_symbols: int = 5):
    """
    Plot l'évolution d'une feature pour plusieurs symbols
    """
    f = plt.figure(figsize=(15, 8))

    for symbol in list(symbol_data.keys())[:n_symbols]:
        data = symbol_data[symbol].sort_values('date_id')
        plt.plot(data['date_id'], data[feature_name],
                 label=f'Symbol {symbol}', alpha=0.7)

    plt.title(f'Évolution de {feature_name} par Symbol')
    plt.xlabel('Date ID')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True)
    plt.show()
    f.savefig(FIGS_DIR / f'feature_evolution_{n_symbols}.png')

