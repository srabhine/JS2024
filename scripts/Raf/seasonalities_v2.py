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

from data_lib.variables import TARGET
from io_lib.paths import LAGS_FEATURES_TRAIN
from math_lib.core import r2_zero
from plot_lib.seasonalities import plot_periodogram

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
# sym = 1
sym = 10
df = df[df['symbol_id'] == sym]
y = df[TARGET]

# Create time stamps
create_time_steps = False

if create_time_steps:
    n_gaps = len(df['time_id'].unique())
    gaps = np.linspace(0, 24*3600 - 10, n_gaps).astype(int)
    dates = pd.date_range('1/1/2019',
                          periods=len(df['date_id'].unique()),
                          freq='D')
    times_all = []
    for d in dates:
        times_all.extend([d + pd.Timedelta(gap, unit="s") for gap in
                          gaps])
    assert len(times_all) == len(df)

    y.index = times_all


# Recompute FFT with normalization
n = len(y)
num_times = len(df['time_id'].unique())
fft_result = np.fft.fft(y) / n  # Normalize by the number of data
# points
magnitude = np.abs(fft_result)


f, axs = plt.subplots(2, figsize=(8, 10))
ix_range = np.arange(1, n//2)
axs[0].plot(ix_range, magnitude[ix_range], '.')
axs[0].set(title='FFT',
           xlabel='Period', ylabel='Amplitude')
ix_range = np.arange(1, 250)
axs[1].set(title='FFT (Zoomed)',
           xlabel='Period', ylabel='Amplitude')
axs[1].plot(ix_range, magnitude[ix_range], '.')
plt.show()

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
t = np.arange(n)
y_hat = (amplitude * np.cos(phase) * np.cos(omega * t) -
                         amplitude * np.sin(phase) * np.sin(omega * t))

# Plot the original and reconstructed signals
plt.figure(figsize=(14, 6))
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t, y_hat, label="Reconstructed",
         lw=3)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Original Series and "
          "Reconstructed Seasonal Component")
plt.grid()
plt.show()

# amplitude, phase

r2w = r2_zero(y, y_hat)
print(r2w)

weights = df['weight']

r2w = r2_zero(y, y_hat, weights)
print(r2w)

