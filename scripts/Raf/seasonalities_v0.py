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

base = pd.Timestamp("2020-1-1")
gen = np.random.default_rng()
gaps = np.cumsum(gen.integers(0, 1800, size=1000))
times = [base + pd.Timedelta(gap, unit="s") for gap in gaps]
index = pd.DatetimeIndex(pd.to_datetime(times))

cal_fourier_gen = CalendarFourier("D", 2)
cal_fourier_gen.in_sample(index)

# Create time stamps
n_gaps = len(df['time_id'].unique())
gaps = np.linspace(0, 24*3600 - 10, n_gaps).astype(int)
dates = pd.date_range('1/1/2019', periods=len(df['date_id'].unique()),
                      freq='D')
times_all = []
for d in dates:
    times_all.extend([d + pd.Timedelta(gap, unit="s") for gap in
                      gaps])

assert len(times_all) == len(df)

target.index = times_all

cal_fourier_gen = CalendarFourier("s", 2)

# Investigate seasonalities

freqs, spectrum = periodogram(target, detrend='linear',
                              window="boxcar", scaling='spectrum')
# freqs, spectrum = signal.periodogram(x, fs)
plt.semilogy(freqs, spectrum)
# plt.ylim([1e-7, 1e2])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
plt.show()

# I could fit on residuals, but since I am using the same process, I can do
# all in one go

# y_train, y_test = y.loc[t_train], y.loc[t_test]

cal_fourier_gen = CalendarFourier('h', 2)
X = cal_fourier_gen.in_sample(target.index)

f, ax = plt.subplots()
ax.plot(X.iloc[:900, 0])
plt.show()


# fourier = CalendarFourier(freq='M', order=2)
# fourier = CalendarFourier(freq='A', order=2)
dp_seas = DeterministicProcess(
    index=times_all,
    constant=True,
    order=1,
    seasonal=False,
    additional_terms=[cal_fourier_gen],
    drop=True,
)
