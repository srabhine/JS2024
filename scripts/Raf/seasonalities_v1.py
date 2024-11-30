"""

@author: Raffaele M Ghigliazza
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import polars as pl
from scipy.signal import periodogram

from statsmodels.tsa.deterministic import CalendarFourier, \
    DeterministicProcess

from data_lib.variables import TARGET
from io_lib.paths import LAGS_FEATURES_TRAIN
from plot_lib.seasonalities import plot_periodogram

# Load data
df = pl.scan_parquet(LAGS_FEATURES_TRAIN).collect().to_pandas()
sym = 1
df = df[df['symbol_id'] == sym]
target = df[TARGET]

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

    target.index = times_all

num_days = 5
num_times = 968
t = np.arange(num_days * num_times)
tau = num_times // 4
x = np.sin( 2 * np.pi / tau * t)
np.random.seed(1234)
e = 0.1 * np.random.randn(len(t))
y = x + e
ds = pd.Series(y)

f, ax = plt.subplots()
ax.plot(y)
plt.show()


# # Chat
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Provided data generation process
# num_days = 5
# num_times = 968
# t = np.arange(num_days * num_times)
# tau = num_times // 4
# x = np.sin(2 * np.pi / tau * t)
# np.random.seed(1234)
# e = 0.1 * np.random.randn(len(t))
# y = x + e
# ds = pd.Series(y)

# Compute FFT
n = len(y)
fft_result = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(y), d=1)  # Default d=1 (sampling interval)
frequencies.min()
frequencies.max()

f, ax = plt.subplots()
ax.plot(frequencies)
plt.show()


# Identify peak frequency (ignoring zero-frequency)
magnitude = np.abs(fft_result)
peak_frequency_idx = np.argmax(magnitude[1:]) + 1  # Skip the zero-frequency term
peak_frequency = frequencies[peak_frequency_idx]
# this matches 1/tau = peak_frequency

# Reconstruct seasonal component
reconstructed_seasonal = np.real(
    fft_result[peak_frequency_idx] * np.exp(2j * np.pi * peak_frequency * t / len(t))
)
a = abs(fft_result[peak_frequency_idx])
reconstructed_seasonal = a * np.sin(2*np.pi*peak_frequency*t)

# Plot original series and reconstructed seasonal component
plt.figure(figsize=(14, 6))
# plt.plot(t, y, label="Original Series", alpha=0.7)
plt.plot(t, reconstructed_seasonal, label="Reconstructed Seasonal Component", alpha=0.7)
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.title("Original Series and Reconstructed Seasonal Component")
# plt.grid()
plt.show()


# Correction:


# Set y = x (clean sinusoidal signal)
y = x

# Recompute FFT with normalization
n = len(y)
fft_result = np.fft.fft(y) / n  # Normalize by the number of data
# points
frequencies = np.fft.fftfreq(n, d=1)

# Identify peak frequency (ignoring zero-frequency)
magnitude = np.abs(fft_result)
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
reconstructed_sin_cos = (amplitude * np.cos(phase) * np.cos(omega * t) -
                         amplitude * np.sin(phase) * np.sin(omega * t))

# Plot the original and reconstructed signals
plt.figure(figsize=(14, 6))
plt.plot(t, y, label="Original Series (y=x)", alpha=0.7)
plt.plot(t, reconstructed_sin_cos, label="Reconstructed Seasonal Component", alpha=0.7, linestyle='--')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Original Series and Corrected Reconstructed Seasonal Component")
plt.grid()
plt.show()

amplitude, phase
