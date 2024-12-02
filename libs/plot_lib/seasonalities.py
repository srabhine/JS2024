"""

To place legend outside
f = plt.figure()
f.subplots_adjust(right=0.5)
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')

@author: Raffaele Ghigliazza
"""
from typing import Optional, Any, Dict, List

import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_pacf

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

from math_lib.math import periodogram_wrapper

PLOT_PARAMS_PD = dict(color='0.75', style='.-', markeredgecolor='0.25',
                      markerfacecolor='0.25', legend=False,)
GRAY_DARK = 0.75*np.ones(3)
GRAY_LIGHT = 0.25*np.ones(3)
PLOT_PARAMS_NP = dict(c=GRAY_DARK,
                      ls='-',
                      marker='.',
                      markeredgecolor=GRAY_LIGHT,
                      markerfacecolor=GRAY_LIGHT)
PARAMS_HELPER = dict(ls='-.', c='k', lw=0.5)


def plot_ts_and_periodogram(y_daily: pd.Series,
                            fs: Optional[float] = None,
                            figsize=(8, 6)):
    f = plt.figure(figsize=figsize)
    ax = subplot(211, title=y_daily.name)
    y_daily.plot(**PLOT_PARAMS_NP, ax=ax)
    ax = subplot(212)
    _, freqs, spectrum = plot_periodogram(y_daily=y_daily, fs=fs, ax=ax)
    return f, freqs, spectrum


def plot_periodogram(y_daily,
                     fs: Optional[float] = None,
                     ax: Optional[Any] = None):
    y_daily = y_daily.ravel()
    if fs is None:
        fs = pd.Timedelta('1Y') / pd.Timedelta('1D')  # 365.2425
        freq = 'daily'
    elif fs == 12:
        freq = 'monthly'
    else:
        freq = 'unspecified'
    # freqs, spectrum = periodogram(y_daily, fs=fs, detrend='linear',
    #                               window="boxcar", scaling='spectrum')
    # # Rem: this is also linear-detrending the series
    freqs, spectrum = periodogram_wrapper(y_daily, fs=fs)

    if ax is None:
        _, ax = plt.subplots()
    markerline, stemlines, baseline = \
        ax.stem(freqs, spectrum, '.', linefmt='grey')
    markerline.set_markersize(3)
    baseline.set_color('none')
    stemlines.set_linewidth(0.5)
    ax.set_xscale('log')
    if freq == 'daily':
        ax.set_xticks([1, 2, 4, 6, 12, fs / 15, fs / 7, fs / 3.5, fs])
        ax.set_xticklabels(['1Y', '6M', '3M', '2M', '1M',
                            '2w', '1w', '3.5d', '1d'])
    elif freq == 'monthly':
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(['1M', '2w', '1w', '0.5w'])
    else:
        ax.set_xticks([1, 2, 3, 4, 6, 12])
        ax.set_xticklabels(ax.get_xticks())
    return ax, freqs, spectrum


def plot_compare_periodofit(y, y_detrended,
                            fs: Optional[float] = None):
    f, axs = plt.subplots(2, 1, sharey=True)
    plot_periodogram(y, fs=fs, ax=axs[0])
    plot_periodogram(y_detrended, fs=fs, ax=axs[1])
    return f


def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    # from matplotlib.offsetbox import AnchoredText

    # Lag x
    x_ = x.shift(lag)

    # Standardize
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()

    # Explained variable: y
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:  # x ix x lagged and y is the original x
        y_ = x

    # Correlation
    corr = y_.corr(x_)

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws, line_kws = dict(alpha=0.75, s=3,), dict(color='C3', )
    ax = sns.regplot(x=x_, y=y_, scatter_kws=scatter_kws, line_kws=line_kws,
                     lowess=True, ax=ax, **kwargs)
    at = AnchoredText(f"{corr:.2f}", prop=dict(size="large"),
                      frameon=True, loc="upper left")
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    if lag < 0:
        ax.set(title=f"Lead {lag}", xlabel=x_.name)
    elif lag > 0:
        ax.set(title=f"Lag {lag}", xlabel=x_.name)

    return ax


def plot_lead_lags(x, y=None, leads=None, lags=6, nrows=1, lagplot_kwargs={},
                   **kwargs):
    import math
    if leads is None:
        lead_lags = lags
        no_leads = True
    else:
        lead_lags = leads + 1 + lags
        no_leads = False
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lead_lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if not no_leads:
            if k < leads:
                ax = lagplot(x, y, lag=-(leads - k), ax=ax, **lagplot_kwargs)
                ax.set_title(f"Lead {-(leads - k)}", fontdict=dict(fontsize=14))
                ax.set(xlabel="", ylabel=f"Lagged {k + 1}")
            elif k == leads and not no_leads:
                ax = lagplot(x, y, lag=0, ax=ax, **lagplot_kwargs)
                ax.set_title(f"Contemporaneous", fontdict=dict(fontsize=14))
                ax.set(xlabel="", ylabel="Lagged 0")
            else:
                ax = lagplot(x, y, lag=k - leads, ax=ax, **lagplot_kwargs)
                ax.set_title(f"Lag {k - leads}", fontdict=dict(fontsize=14))
                ax.set(xlabel="", ylabel=f"Lagged {k - leads}")
        else:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel=f"Lagged {k + 1}")

    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


def plot_detect_seasonalities(y: pd.Series, y_resid: pd.Series):
    plot_ts_and_periodogram(y)
    plot_compare_periodofit(y, y_resid)


def plot_detect_lags(y: pd.Series, y_resid: pd.Series,
                     lags: int = 12, nrows: int = 2):
    plot_pacf(y, lags=lags, method='ywm')
    plot_lead_lags(x=y, lags=lags, nrows=nrows)
    plot_pacf(y_resid, lags=lags, method='ywm')


def plot_ts_fit(y: pd.Series, y_fit: pd.Series,
                y_pred: Optional[pd.Series] = None, ax: Optional[Any] = None):
    if ax is None:
        _, ax = plt.subplots()
    y.plot(**PLOT_PARAMS_PD)
    y_fit.plot()
    if y_pred is not None:
        y_pred.plot()
    return ax


