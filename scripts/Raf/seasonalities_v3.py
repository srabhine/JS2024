"""

@author: Raffaele M Ghigliazza
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import polars as pl
from IPython.core.pylabtools import figsize
from scipy.signal import periodogram

from statsmodels.tsa.deterministic import CalendarFourier, \
    DeterministicProcess

from data_lib.variables import TARGET, SYMBOLS
from io_lib.paths import LAGS_FEATURES_TRAIN
from math_lib.core import r2_zero
from plot_lib.seasonalities import plot_periodogram

# Load data
df_all = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()

predictions, freqs, periods, keys = [], [], [], []

for sym in SYMBOLS:
    print(f'Calculating symbol = {sym} ...')

    df = df_all[df_all['symbol_id'] == sym]
    y = df[TARGET]

    # Recompute FFT with normalization
    n = len(y)
    num_times = len(df['time_id'].unique())
    fft_result = np.fft.fft(y) / n
    magnitude = np.abs(fft_result)

    frequencies = np.fft.fftfreq(n, d=1)

    # Identify peak frequency (ignoring zero-frequency)
    peak_frequency_idx = np.argmax(magnitude[1:]) + 1  # Skip zero-frequency term
    peak_frequency = frequencies[peak_frequency_idx]

    # Fourier coefficient at the peak frequency
    peak_coefficient = fft_result[peak_frequency_idx]

    # Recompute amplitude and phase with normalization
    amplitude = np.abs(peak_coefficient) * 2  # Factor of 2 for real-valued signals
    phase = np.angle(peak_coefficient)

    # Angular frequency for the peak
    omega = 2 * np.pi * peak_frequency

    # Reconstruct the signal as a linear combination of sine and cosine
    t = np.arange(2 * n)
    y_hat = (amplitude * np.cos(phase) * np.cos(omega * t) -
                             amplitude * np.sin(phase) * np.sin(omega * t))
    predictions.append(pd.Series(y_hat))
    freqs.append(peak_frequency)
    periods.append(1/peak_frequency)
    keys.append(sym)


predictions = pd.concat(predictions, keys=keys)

f, ax = plt.subplots()
ax.bar(range(len(periods)), periods)
plt.show()
