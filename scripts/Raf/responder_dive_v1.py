"""

@author: Raffaele M Ghigliazza
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_lib.datasets import get_symbols_dataset
from data_lib.synthetic import generate_ar1
from io_lib.paths import FIGS_DIR
from plot_lib.timeseries import plot_acf_pacf

figs_dir = FIGS_DIR / 'eda'

sym = 1
data_sym = get_symbols_dataset(sym=sym)  # (160688, 89)
data_sym['date_id'] = data_sym.index.get_level_values(0)
data_sym['time_id'] = data_sym.index.get_level_values(1)
data_sym.index = range(len(data_sym))


# Review ACF and PACF
ds = pd.Series(generate_ar1(n=1000, rho=0.9))

f, axs = plot_acf_pacf(ds, figs_dir=figs_dir, fnm='acf_pacf_ar1')

f, axs = plot_acf_pacf(data_sym['responder_6'], figs_dir=figs_dir,
                       fnm='acf_pacf_responder_6')

# Looking at lag
for sym in range(10, 20):
    data_sym = get_symbols_dataset(sym=sym)  # (160688, 89)
    ds = data_sym['responder_6']
    ds_lag = ds.shift(-1)
    df = pd.concat([ds, ds_lag], axis=1,
                   keys=['lag', 'target']).dropna()
    print(f'{sym}')
    print(np.corrcoef(df['lag'].values, df['target'].values))

    f, ax = plt.subplots()
    f.suptitle(f'{sym}')
    ax.scatter(df['lag'], df['target'], alpha=0.01)
    ax.set(title='Responder vs Its Lag', xlabel='lag 6',
           ylabel='responder 6')
    f.savefig(figs_dir / f'{sym}_lag.png')
    plt.close(f)
