import numpy as np
from scipy.signal import lombscargle
import matplotlib.pyplot as plt

# Define frequencies to test
freqs_rps = np.linspace(0.1, 15.0*2*np.pi, 500)

data_file_clean = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst_soc2p2_hi_lo_chg.csv'
data_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)

# Compute the Lomb-Scargle periodogram
power_VoVca = lombscargle(data_raw.time, data_raw.VoVca, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_VoVca)]/(2*np.pi)
print(f"Dominant frequency Ib Amp: {dominant_frequency:.2f} Hz")

power_VoVcn = lombscargle(data_raw.time, data_raw.VoVcn, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_VoVcn)]/(2*np.pi)
print(f"Dominant frequency Ib Noa: {dominant_frequency:.2f} Hz")

power_Tbv = lombscargle(data_raw.time, data_raw.Tb, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Tbv)]/(2*np.pi)
print(f"Dominant frequency Tbv: {dominant_frequency:.2f} Hz")

power_Vbv = lombscargle(data_raw.time, data_raw.Vb, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Vbv)]/(2*np.pi)
print(f"Dominant frequency Vbv: {dominant_frequency:.2f} Hz")


plt.figure(figsize=(10, 6))
plt.plot(freqs_rps / (2 * np.pi), power_VoVca, color='black', linestyle='-', label='Vo-Vc amp at ADC, Volts') # Convert to Hz for x-axis
plt.plot(freqs_rps / (2 * np.pi), power_VoVcn, color='orange', linestyle='--', label='Vo-Vc noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.grid(True)
plt.figure(figsize=(10, 6))
plt.plot(freqs_rps / (2 * np.pi), power_Tbv, color='red', linestyle='-', label='Tbv at ADC, Volts') # Convert to Hz for x-axis
plt.plot(freqs_rps / (2 * np.pi), power_Vbv, color='blue', linestyle='--', label='Vbv at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.grid(True)
plt.show()