import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# Load the CSV file
filename = 'C:\\Users\\Lenovo\\Desktop\\D^2_ติวเตอร์\\vernier\\C5.csv'
data = pd.read_csv(filename)

# Check for expected columns
if 'Data Set 1:Time(s)' not in data.columns or 'Data Set 1:Sound Pressure' not in data.columns:
    raise ValueError("CSV file must contain 'Data Set 1:Time(s)' and 'Data Set 1:Sound Pressure' columns.")

# Extract time and amplitude values
time = data['Data Set 1:Time(s)'].values
amplitude = data['Data Set 1:Sound Pressure'].values

# Number of samples
N = len(time)

# Check if we have enough samples
if N <= 1:
    raise ValueError("Not enough data to perform FFT.")

# Sampling frequency (inverse of the mean time difference)
T = np.mean(np.diff(time))  # Time between samples

# Perform FFT
yf = fft(amplitude)
xf = fftfreq(N, T)[:N // 2]  # Frequency axis
amplitude_spectrum = 2.0 / N * np.abs(yf[:N // 2])  # Magnitude of the FFT

# Find peaks in the amplitude spectrum
peaks, _ = find_peaks(amplitude_spectrum, height=0.01)  # Adjust height threshold as needed

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the amplitude spectrum
plt.subplot(2, 1, 1)
plt.plot(xf, amplitude_spectrum, label='Amplitude Spectrum',color='red')
plt.scatter(xf[peaks], amplitude_spectrum[peaks], color='y', label='Peaks')  # Show peaks with a label
plt.title('Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()  # Ensure that the legend is called after all labels are set

# Plot the waveform
plt.subplot(2, 1, 2)
plt.plot(time, amplitude, label='Waveform')
plt.title('Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()  # Include legend in time domain plot if necessary

plt.tight_layout()
plt.show()


# Print peak frequencies
peak_frequencies = xf[peaks]
print("Peak Frequencies (Hz):", peak_frequencies)
