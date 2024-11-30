"""

@author: Raffaele Ghigliazza
"""
import time
from typing import Optional, Union

import numpy as np
import pandas as pd
# from math_lib.probability import kde_fit_empirical

from scipy.signal import periodogram
from statsmodels.tsa.deterministic import DeterministicProcess


def normalize(x, loc: Union[float, np.ndarray] = 0.0,
              scale: Union[float, np.ndarray] = 1.0):
    return (x - loc) / scale


def denormalize(x, loc: Union[float, np.ndarray] = 0.0,
                scale: Union[float, np.ndarray] = 1.0):
    return x * scale + loc


def denormalize_moments(mu_norm, sg_norm,
                        loc: Union[float, np.ndarray] = 0.0,
                        scale: Union[float, np.ndarray] = 1.0):
    mu = denormalize(mu_norm, loc=loc, scale=scale)
    sg = denormalize(sg_norm, loc=0, scale=scale)
    return mu, sg


def periodogram_wrapper(y_daily, fs: Optional[float] = None):
    y_daily = y_daily.ravel()
    if fs is None:
        fs = pd.Timedelta('1Y') / pd.Timedelta('1D')  # 365.2425
        freq = 'daily'
    elif fs == 12:
        freq = 'monthly'
    else:
        freq = 'unspecified'
    freqs, spectrum = periodogram(y_daily, fs=fs, detrend='linear',
                                  window="boxcar", scaling='spectrum')
    # Rem: this is also linear-detrending the series
    return freqs, spectrum


def normalize_weights(w):
    return w / w.sum()


def seconds2hours(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def calc_fading_weights(i_curr, lam: float = 1e-3):
    sample_weight_decay = np.exp(-lam * (np.arange(i_curr + 1, 0, -1) - 1))
    return sample_weight_decay


# def calc_memory_weights(x: np.ndarray, ratio: float = 10):
#     x = x.reshape(-1, 1)
#     pdf, x_supp, kde, h, bin_centers, bin_edges = kde_fit_empirical(x)
#     log_probs_org = kde.score_samples(x)
#     probs_org = np.exp(log_probs_org)
#     l_min, l_max = probs_org.min(), probs_org.max()
#     f_min, f_max = 1, ratio
#     weight_mapped = f_min + (f_max - f_min) / \
#                     (l_max - l_min) * (probs_org - l_min)
#     weights_freq = 1 / weight_mapped
#     weights_freq /= weights_freq.sum()
#     # Check: weights_freq.sum()
#     return weights_freq


def calc_tot_return_scalar(pnl_base: np.ndarray,
                           pnl_target: np.ndarray):
    return pnl_base.cumsum()[-1] / pnl_target.cumsum(axis=0)[-1, :]
