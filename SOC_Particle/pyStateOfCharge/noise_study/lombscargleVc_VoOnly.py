import numpy as np
from scipy.signal import lombscargle
import matplotlib.pyplot as plt

# Obtain data by:  Cx20000.  Save as 'burst'  Data is approximately 40 Hz so Nyquist is 20 Hz = 125 r/s
# handedit file to remove any information above header row and below end
# primary peak at 1.6 Hz excites sample mode resolution artifacts at 1.6 +  3.1*n

# Define frequencies to test
# data_file_clean = 'burstVo_Vc_Cx20000_soc2p2_hi_lo_chg.csv'
# data_file_clean = 'burstVo_Vc_Cx20000_BTremo_soc2p2_hi_lo_chg.csv'
# data_file_clean = 'burstForKF_soc2p2_hi_lo_chg.csv'  # just to look at data
data_file_clean = 'burstVo_Vc_Cx20000_Bare_soc2p2_hi_lo_chg.csv'

# Get data and statistics
data_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
sampt = []
sampf = []
for i in range(len(data_raw.time)):
    if i == 0:
        sampt.append(0.)
        sampf.append(0.)
    else:
        sampt.append(data_raw.time[i] - data_raw.time[i-1])
        sampf.append(1./(data_raw.time[i] - data_raw.time[i-1]))
sample_freq_hz = np.average(sampf)
nyquist = sample_freq_hz
N = len(data_raw)
freqs_rps = np.linspace(0.1, nyquist*2*np.pi, N)

# Compute the Lomb-Scargle periodogram
power_Vcn = lombscargle(data_raw.time, data_raw.Vcn, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Vcn)]/(2*np.pi)
print(f"Dominant frequency Vc: {dominant_frequency:.2f} Hz")

power_Von = lombscargle(data_raw.time, data_raw.Von, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Von)]/(2*np.pi)
print(f"Dominant frequency Vc: {dominant_frequency:.2f} Hz")

power_Vo_Vcn = lombscargle(data_raw.time, data_raw.VoVcn, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Vo_Vcn)]/(2*np.pi)
print(f"Dominant frequency Vc: {dominant_frequency:.2f} Hz")

# Time history
plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, data_raw.Von, color='red', linestyle='--', label='Vo noa at ADC, Volts') # Convert to Hz for x-axis
plt.plot(data_raw.time, data_raw.Vcn, color='green', linestyle='-', label='Vc noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('Volts')
plt.title('Time History of Data Used Periodogram ' + data_file_clean)
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, data_raw.VoVcn, color='red', linestyle='--', label='Vo-Vc Noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('Volts')
plt.title('Time History of Data Used Periodogram ' + data_file_clean)
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, sampt, color='green', linestyle='-', label='Vc Sample Time, sec') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('sec')
plt.title('Time History of Sample Time Data Used Periodogram ' + data_file_clean)
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, sampf, color='green', linestyle='-', label='Sample Vc frequency, Hz') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('Hz')
plt.title('Time History of Sample Frequency Data Used Periodogram ' + data_file_clean)
plt.grid(True)

# Periodograms
plt.figure(figsize=(10, 6))
plt.semilogx(freqs_rps / (2 * np.pi), power_Vcn, color='blue', linestyle='--', label='Vc noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram ' + data_file_clean)
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.semilogx(freqs_rps / (2 * np.pi), power_Von, color='magenta', linestyle='-', label='Vo noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram ' + data_file_clean)
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.semilogx(freqs_rps / (2 * np.pi), power_Vo_Vcn, color='magenta', linestyle='-', label='Vo-Vc noa at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram ' + data_file_clean)
plt.grid(True)

plt.show()
