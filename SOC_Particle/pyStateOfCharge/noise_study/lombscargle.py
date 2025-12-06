import numpy as np
from scipy.signal import lombscargle
import matplotlib.pyplot as plt

# Obtain data by:  Cx2500 (Cx10000 for build of Vca only).  Save as 'burst'  Data is approximately 40 Hz so Nyquist is 20 Hz = 125 r/s
# handedit file to remove any information above header row
# Results are meaningless without running the lombscargleVcOnly.py file and reviewing notes in that file

# Define frequencies to test
freqs_rps = np.linspace(0.1, 40.0*2*np.pi, 500)

# data_file_clean = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst_soc2p2_hi_lo_chg.csv'
# data_file_clean = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burstCapNoa_soc2p2_hi_lo_chg.csv'
# data_file_clean = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burstCapNoa104_soc2p2_hi_lo_chg.csv'
# data_file_clean = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burstRebase_soc2p2_hi_lo_chg.csv'
data_file_clean = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst_forced0V+V-_soc2p2_hi_lo_chg.csv'

data_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)

# Compute the Lomb-Scargle periodogram
power_Von = lombscargle(data_raw.time, data_raw.Von, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Von)]/(2*np.pi)
print(f"Dominant frequency Ib Amp: {dominant_frequency:.2f} Hz")

power_Voa = lombscargle(data_raw.time, data_raw.Voa, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Voa)]/(2*np.pi)
print(f"Dominant frequency Ib Amp: {dominant_frequency:.2f} Hz")

power_VoVca = lombscargle(data_raw.time, data_raw.VoVca, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_VoVca)]/(2*np.pi)
print(f"Dominant frequency Ib Amp: {dominant_frequency:.2f} Hz")

power_VoVcn = lombscargle(data_raw.time, data_raw.VoVcn, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_VoVcn)]/(2*np.pi)
print(f"Dominant frequency Ib Noa: {dominant_frequency:.2f} Hz")

power_Tbv = lombscargle(data_raw.time, data_raw.Tbv, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Tbv)]/(2*np.pi)
print(f"Dominant frequency Tbv: {dominant_frequency:.2f} Hz")

power_Vbv = lombscargle(data_raw.time, data_raw.Vbv, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Vbv)]/(2*np.pi)
print(f"Dominant frequency Vbv: {dominant_frequency:.2f} Hz")

# Time history
plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, data_raw.Voa, color='magenta', linestyle='-', label='Vo amp at ADC, Volts') # Convert to Hz for x-axis
plt.plot(data_raw.time, data_raw.Von, color='red', linestyle='--', label='Vo noa at ADC, Volts') # Convert to Hz for x-axis
plt.plot(data_raw.time, data_raw.Vcn-0.05, color='blue', linestyle='--', label='Vc noa -0.05 at ADC, Volts') # Convert to Hz for x-axis
plt.plot(data_raw.time, data_raw.Vca-0.05, color='green', linestyle='-', label='Vc amp -0.05 at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('Volts')
plt.title('Time History of Data Used Periodogram')
plt.grid(True)
plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, data_raw.Vbv, color='orange', linestyle='--', label='Vb at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Volts')
plt.title('Time History of Data Used Periodogram')
plt.grid(True)
plt.grid(True)
plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, data_raw.Tbv, color='black', linestyle='-', label='Tb at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('Volts')
plt.title('Time History of Data Used Periodogram')
plt.grid(True)
plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, data_raw.VoVca, color='magenta', linestyle='-', label='Vo-Vc amp at ADC, Volts') # Convert to Hz for x-axis
plt.plot(data_raw.time, data_raw.VoVcn, color='blue', linestyle='--', label='Vo-Vc noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('Delta V')
plt.title('Time History of Data Used Periodogram')
plt.grid(True)

# Periodograms
plt.figure(figsize=(10, 6))
plt.semilogx(freqs_rps / (2 * np.pi), power_Voa, color='magenta', linestyle='-', label='Vo amp at ADC, Volts') # Convert to Hz for x-axis
plt.semilogx(freqs_rps / (2 * np.pi), power_Von, color='blue', linestyle='--', label='Vo noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.semilogx(freqs_rps / (2 * np.pi), power_VoVca, color='magenta', linestyle='-', label='Vo-Vc amp at ADC, Volts') # Convert to Hz for x-axis
plt.semilogx(freqs_rps / (2 * np.pi), power_VoVcn, color='blue', linestyle='--', label='Vo-Vc noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.semilogx(freqs_rps / (2 * np.pi), power_Tbv, color='red', linestyle='-', label='Tbv at ADC, Volts') # Convert to Hz for x-axis
plt.semilogx(freqs_rps / (2 * np.pi), power_Vbv, color='blue', linestyle='--', label='Vbv at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.grid(True)
plt.show()