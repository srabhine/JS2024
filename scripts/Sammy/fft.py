import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import polars as pl


from libs.io_lib.paths import LAGS_FEATURES_TRAIN, LAGS_FEATURES_VALID
from libs.one_big_lib import stack_features_by_sym

#feature_names = ([f"feature_{i:02d}" for i in range(79)]
#                 + [f"feature_{i:02d}_lag_1" for i in range(79)]
#                 + [f"responder_{idx}_lag_1" for idx in range(9)])

def load_data(start_id, end_id):
   folder_paths = [
       f"{LAGS_FEATURES_TRAIN}/date_id={date_id}/00000000.parquet"
       for date_id in range(start_id, end_id + 1)
   ]
   
   lazy_frames = [pl.scan_parquet(path) for path in folder_paths]
   combined_data = pl.concat(lazy_frames).collect().to_pandas()
   
   return combined_data.ffill().fillna(0)


data_train = load_data(start_id=1544, end_id=1644)

# Filter for symbol_id = 1
#data_train = data_train[data_train['symbol_id'] == 1]

# Get the responder series
responder_series = data_train['responder_6'].values
n_samples = len(responder_series)

# Fourier transform calculations
fft_coeffs = fftpack.fft(responder_series)
frequencies = fftpack.fftfreq(n_samples)
power_spectrum = np.abs(fft_coeffs)**2

# Filter for positive frequencies
positive_freq_mask = frequencies > 0
frequencies = frequencies[positive_freq_mask]
power_spectrum = power_spectrum[positive_freq_mask]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(frequencies, power_spectrum)
plt.xlabel('Frequency (cycles per sample)')
plt.ylabel('Power')
plt.title('Power Spectrum of Responder 6')
plt.grid(True)
plt.show()

# Print some information about the data
print(f"Number of samples analyzed: {n_samples}")
print(f"Time period covered: {data_train['date_id'].min()} to {data_train['date_id'].max()}")



#### Seasonalities ####


# Calculate periods from frequencies
periods = 1/frequencies  # Convert frequencies to periods (number of samples)

# Find the most significant peaks
peak_indices = np.argsort(power_spectrum)[-10:]  # Get indices of top 10 peaks
significant_periods = periods[peak_indices]
significant_powers = power_spectrum[peak_indices]

# Print the most significant periods
print("\nMost significant periods (in number of samples):")
for period, power in zip(significant_periods, significant_powers):
    print(f"Period of {period:.2f} samples with power: {power:.2e}")

# Create a more interpretable plot
plt.figure(figsize=(12, 6))
plt.plot(periods, power_spectrum)
plt.xlabel('Period (number of samples)')
plt.ylabel('Power')
plt.title('Power Spectrum of Responder 6 by Period')
plt.xscale('log')  # Log scale makes patterns easier to see
plt.grid(True)
plt.show()