import numpy as np
from scipy.signal import lombscargle
import matplotlib.pyplot as plt

# Obtain data by:  Cx2500 (Cx10000 for build of Vca only).  Save as 'burst'  Data is approximately 40 Hz so Nyquist is 20 Hz = 125 r/s
# handedit file to remove any information above header row
# primary peak at 1.6 Hz excites sample mode resolution artifacts at 1.6 +  3.1*n

# Define frequencies to test
# data_file_clean = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burstVcCx10000_soc2p2_hi_lo_chg.csv'
data_file_clean = 'burstVcCx10000_soc2p2_hi_lo_chg.csv'
data_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
nyquist = 92.5/2.  # get this by looking at plots and rerunning
numpts = len(data_raw)
freqs_rps = np.linspace(0.1, nyquist*2*np.pi, 20000)

# Compute the Lomb-Scargle periodogram
power_Vca = lombscargle(data_raw.time, data_raw.Vca, freqs_rps, floating_mean=True)
dominant_frequency = freqs_rps[np.argmax(power_Vca)]/(2*np.pi)
print(f"Dominant frequency Vc: {dominant_frequency:.2f} Hz")

# Time history
plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, data_raw.Vca, color='green', linestyle='-', label='Vc amp at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('Volts')
plt.title('Time History of Data Used Periodogram')
plt.grid(True)

sampt = []
sampf = []
for i in range(len(data_raw.time)):
    if i == 0:
        sampt.append(0.)
        sampf.append(0.)
    else:
        sampt.append(data_raw.time[i] - data_raw.time[i-1])
        sampf.append(1./(data_raw.time[i] - data_raw.time[i-1]))


plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, sampt, color='green', linestyle='-', label='Vc Sample Time, sec') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('sec')
plt.title('Time History of Sample Time Data Used Periodogram')
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(data_raw.time, sampf, color='green', linestyle='-', label='Sample Vc frequency, Hz') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Time, s')
plt.ylabel('Hz')
plt.title('Time History of Sample Frequency Data Used Periodogram')
plt.grid(True)

# Periodograms
plt.figure(figsize=(10, 6))
plt.semilogx(freqs_rps / (2 * np.pi), power_Vca, color='blue', linestyle='--', label='Vc amp at ADC, Volts') # Convert to Hz for x-axis
plt.legend(loc=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Lomb-Scargle Periodogram')
plt.grid(True)

plt.show()
