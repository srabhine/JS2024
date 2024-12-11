"""

@author: Raffaele M Ghigliazza
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from one_big_lib import stack_features_by_sym, FEATS, TARGET, SYMBOLS
from data_lib.variables import TARGET, SYMBOLS


# Load data

file_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
start_dt = 1200
end_dt = 1690

data = pl.scan_parquet(file_path
                       ).select(
    pl.int_range(pl.len(), dtype=pl.UInt32).alias("id"),
    pl.all(),
).filter(
    pl.col("date_id").gt(start_dt),
    pl.col("date_id").le(end_dt),
)

data = data.collect().to_pandas()

data.replace([np.inf, -np.inf], 0, inplace=True)
data = data.fillna(0)

df_by_sym = stack_features_by_sym(data)
df_by_sym.head()

df_all = df_by_sym

predictions, freqs, periods, keys = [], [], [], []
df_params = pd.DataFrame(columns=['a', 'om', 'phase'], index=SYMBOLS)

for sym in SYMBOLS:
    print(f'Calculating symbol = {sym} ...')

    # df = df_all[df_all['symbol_id'] == sym]
    y = df_by_sym[(TARGET, sym)]
    y = y.ffill().fillna(0)
    # Recompute FFT with normalization
    n = len(y)
    # num_times = len(df['time_id'].unique())
    fft_result = np.fft.fft(y) / n
    magnitude = np.abs(fft_result)
    magnitude = magnitude[:len(magnitude)//2]

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
    if np.any(np.isnan(y_hat)):
        print(y_hat)


    predictions.append(pd.Series(y_hat))
    freqs.append(peak_frequency)
    periods.append(1/peak_frequency)
    keys.append(sym)
    # r2_zero(y, y_hat, weights)
    df_params.loc[sym, "a"] = amplitude
    df_params.loc[sym, "om"] = omega
    df_params.loc[sym, "phase"] = phase


predictions = pd.concat(predictions, keys=keys, axis=1)

f, ax = plt.subplots()
ax.bar(range(len(periods)), periods)
plt.show()


plt.plot(predictions[30])
plt.show()


y = df_by_sym[(TARGET, 33)]
plt.plot(y.values)
plt.show()

ti = t[-1]+1
y_hat = (amplitude * np.cos(phase) * np.cos(omega * ti) - amplitude * np.sin(phase) * np.sin(omega * ti))


df_params.to_csv('/home/zt/pyProjects/JaneSt/Team/data/params.csv')