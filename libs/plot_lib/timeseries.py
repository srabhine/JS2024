"""

@author: Raffaele M Ghigliazza
"""
from typing import Optional, Any
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

from mat_lib.timeseries import create_lags


def plot_acf_pacf(ds: pd.Series,
                  lags: Optional[Any] = None,
                  figs_dir: Optional[Any] = None,
                  fnm: str = 'acf_pacf'):

    if lags is None:
        lags = np.arange(1, 20)

    df_lags = create_lags(ds, lags).dropna()

    reg = LinearRegression()
    y = df_lags['orig'].values
    acf = pd.Series(index=lags)
    for lag in lags:
        x_acf = df_lags[f'lag_{lag}'].values.reshape(-1, 1)
        reg.fit(x_acf, y)
        acf.loc[lag] = np.sqrt(reg.score(x_acf, y))

    # Plot ACF
    f, axs = plt.subplots(3, figsize=(10, 8))
    axs[0].plot(ds)
    axs[0].set(title='Data')
    axs[1].stem(acf)
    axs[1].set(title='ACF')
    plot_pacf(ds, lags=20, ax=axs[2])
    axs[2].set(title='PACF')
    f.tight_layout()
    if figs_dir is not None:
        f.savefig(figs_dir / f'{fnm}.png')
    plt.close(f)

    return f, axs
