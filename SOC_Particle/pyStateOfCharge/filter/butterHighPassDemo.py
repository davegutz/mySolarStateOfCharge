import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def butter_highpass_filter(data, cutoff_freq, fs, order=5):
    """
    Designs and applies a digital high-pass Butterworth filter to data.

    Parameters:
    data (array-like): The input signal to filter.
    cutoff_freq (float): The frequency above which signals pass (in Hz).
    fs (float): The sampling rate of the signal (in Hz).
    order (int): The order of the filter (default is 5).

    Returns:
    array-like: The filtered signal.
    """
    # Calculate the normalized cutoff frequency (Nyquist frequency is fs/2)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    # Design the filter coefficients (b and a)
    # 'high' specifies a high-pass filter
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    # Apply the filter using filtfilt to avoid phase delay
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


# Example Usage:
if __name__ == "__main__":

    # 1. Define filter parameters
    fs = 1000.0       # Sampling rate (Hz)
    cutoff_freq = 50.0  # Cutoff frequency (Hz)
    order = 5

    # 2. Generate some sample data (a mix of high and low frequencies)
    t = np.linspace(0, 10, int(fs * 10), endpoint=False)  # Time vector
    # Signal 1: 5 Hz low frequency
    # Signal 2: 150 Hz high frequency
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 150 * t) + np.random.normal(0, 0.5, len(t))

    # 3. Apply the high-pass filter
    filtered_data = butter_highpass_filter(data, cutoff_freq, fs, order)

    # 4. (Optional) Plot the results to visualize the effect
    plt.figure(figsize=(10, 6))
    plt.plot(t, data, label='Original signal', alpha=0.5)
    plt.plot(t, filtered_data, label='Filtered signal (HPF)', color='red', linewidth=2)
    plt.title('High-Pass Filter Example')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
