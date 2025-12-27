# BatteryKF - general purpose battery class for embedded KF
# Copyright (C) 2021 Dave Gutz
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# See http://www.fsf.org/licensing/licenses/lgpl.txt for full license text.

"""1x1 General Purpose Kalman Filter.   Inherit from this class and include kf_predict and
kf_update methods in the parent."""

global mon_run
from load_data import write_clean_file
from myFilters import LagTustin, LagExp
from KF1x1 import KF1x1VarDt, KF1x1VarDt

def plot_1(plt=None, mr=None, mv=None, title=None):
    plt.figure()
    plt.subplot(111)
    plt.title(title)
    plq(plt, mr, 'time', mr, 'Voa', add=0.6, color='blue', linestyle='-', label='Voa' + run_str + '+0.6')
    plq(plt, mv, 'time', mv, 'Voa', add=0.6, color='red', linestyle='--', label='Voa' + ver_str + '+0.6')
    plq(plt, mr, 'time', mr, 'Vca', add=0.4, color='blue', linestyle='-', label='Vca' + run_str + '+0.4')
    plq(plt, mv, 'time', mv, 'Vca', add=0.4, color='red', linestyle='--', label='Vca' + ver_str + '+0.4')
    plq(plt, mr, 'time', mr, 'VoVca', add=0.2, color='blue', linestyle='-', label='VoVca' + run_str + '+0.2')
    plq(plt, mv, 'time', mv, 'VoVca', add=0.2, color='red', linestyle='--', label='VoVca' + ver_str + '+0.2')
    plq(plt, mr, 'time', mr, 'Von', add=0.0, color='blue', linestyle='-', label='Von' + run_str + '+0.0')
    plq(plt, mv, 'time', mv, 'Von_kf', add=0.0, color='red', linestyle='--', label='Von_kf' + ver_str + '+0.0')
    plq(plt, mr, 'time', mr, 'Vcn', add=-0.8, color='blue', linestyle='-', label='Vcn' + run_str + '-0.8')
    plq(plt, mv, 'time', mv, 'Vcn', add=-0.8, color='red', linestyle='--', label='Vcn' + ver_str + '-0.8')
    plq(plt, mr, 'time', mr, 'VoVcn', add=-0.0, color='blue', linestyle='-', label='VoVcn' + run_str + '-0.0')
    plq(plt, mv, 'time', mv, 'VoVcn', add=-0.0, color='red', linestyle='--', label='VoVcn' + ver_str + '-0.0')
    plq(plt, mr, 'time', mr, 'Vbv', add=-0.6, color='blue', linestyle='-', label='Vbv' + run_str + '-0.6')
    plq(plt, mv, 'time', mv, 'Vbv', add=-0.6, color='red', linestyle='--', label='Vbv' + ver_str + '-0.6')
    plq(plt, mr, 'time', mr, 'Tbv', add=-0.8, color='blue', linestyle='-', label='Tbv' + run_str + '-0.8')
    plq(plt, mv, 'time', mv, 'Tbv', add=-0.8, color='red', linestyle='--', label='Tbv' + ver_str + '-0.8')
    plt.legend(loc=1)
    return plt

def plot_2(plt=None, mr=None, mv=None, title=None):
    plt.figure()
    plt.subplot(111)
    plt.title(title)
    plq(plt, mv, 'time', mv, 'VoVcn', color='red', linestyle='-', label='VoVcn' + ver_str)
    plq(plt, mv, 'time', mv, 'VoVcnA', color='green', linestyle='--', label='VoVcnA' + ver_str)
    plt.legend(loc=1)
    return plt

def plot_3(plt=None, mr=None, mv=None, title=None):
    plt.figure()
    plt.subplot(111)
    plt.title(title)
    plq(plt, mr, 'time', mr, 'Voa', add=0.0, color='blue', linestyle='-', label='Voa' + run_str + '+0.0')
    plq(plt, mv, 'time', mv, 'Voa', add=0.0, color='red', linestyle='--', label='Voa' + ver_str + '+0.0')
    plq(plt, mr, 'time', mr, 'Vca', add=-0.2, color='blue', linestyle='-', label='Vca' + run_str + '-0.2')
    plq(plt, mv, 'time', mv, 'Vca', add=-0.2, color='red', linestyle='--', label='Vca' + ver_str + '-0.2')
    plq(plt, mr, 'time', mr, 'VoVca', add=-0.4, color='blue', linestyle='-', label='VoVca' + run_str + '-0.4')
    plq(plt, mv, 'time', mv, 'VoVca', add=-0.4, color='red', linestyle='--', label='VoVca' + ver_str + '-0.4')
    plt.legend(loc=1)
    return plt

def plot_4(plt=None, mr=None, mv=None, title=None):
    plt.figure()
    plt.subplot(231)
    plt.title(title)
    plq(plt, mr, 'time', mr, 'Von', add=0.0, color='blue', linestyle='-', label='Von' + run_str + '+0.0')
    plq(plt, mv, 'time', mv, 'Von_kf', add=0.0, color='red', linestyle='--', label='Von_kf' + ver_str + '+0.0')
    plt.legend(loc=1)
    plt.subplot(234)
    plq(plt, mr, 'time', mr, 'Voa', add=0.0, color='blue', linestyle='-', label='Voa' + run_str + '+0.0')
    plq(plt, mv, 'time', mv, 'Voa', add=0.0, color='red', linestyle='--', label='Voa' + ver_str + '+0.0')
    plt.legend(loc=1)
    plt.subplot(232)
    plq(plt, mr, 'time', mr, 'Von_rms', add=0.0, color='blue', linestyle='-', label='Von_rms' + run_str + '+0.0')
    plq(plt, mv, 'time', mv, 'Von_rms', add=0.0, color='red', linestyle='--', label='Von_rms' + ver_str + '+0.0')
    plt.legend(loc=1)
    plt.subplot(235)
    plq(plt, mr, 'time', mr, 'Voa_rms', add=0.0, color='blue', linestyle='-', label='Voa_rms' + run_str + '+0.0')
    plq(plt, mv, 'time', mv, 'Voa_rms', add=0.0, color='red', linestyle='--', label='Voa_rms' + ver_str + '+0.0')
    plt.legend(loc=1)
    plt.subplot(233)
    plq(plt, mv, 'time', mv, 'Von_atten', add=0.0, color='blue', linestyle='-', label='Von_atten' + run_str + '+0.0')
    plt.legend(loc=1)
    plt.subplot(236)
    plq(plt, mv, 'time', mv, 'Voa_atten', add=0.0, color='blue', linestyle='-', label='Voa_atten' + run_str + '+0.0')
    plt.legend(loc=1)
    return plt

def plot_5(plt=None, mr=None, mv=None, title=None):
    plt.figure()
    plt.subplot(111)
    plt.title(title)
    plq(plt, mr, 'time', mr, 'Vbv', add=-0.6, color='blue', linestyle='-', label='Vbv' + run_str + '-0.6')
    plq(plt, mv, 'time', mv, 'Vbv', add=-0.6, color='red', linestyle='--', label='Vbv' + ver_str + '-0.6')
    plq(plt, mr, 'time', mr, 'Tbv', add=-0.8, color='blue', linestyle='-', label='Tbv' + run_str + '-0.8')
    plq(plt, mv, 'time', mv, 'Tbv', add=-0.8, color='red', linestyle='--', label='Tbv' + ver_str + '-0.8')
    plt.legend(loc=1)
    return plt

def plot_6(plt=None, mr=None, mv=None, title=None):
    plt.figure()
    plt.subplot(111)
    plt.title(title)
    plq(plt, mr, 'time', mr, 'VoVcn', add=-0.0, color='blue', linestyle='-', label='VoVcn' + run_str + '-0.0')
    plq(plt, mv, 'time', mv, 'VoVcn', add=-0.0, color='red', linestyle='--', label='VoVcn' + ver_str + '-0.0')
    plq(plt, mv, 'time', mv, 'VofVcfn', add=-0.0, color='cyan', linestyle='-.', label='VofVcfn' + ver_str + '-0.0')
    plt.legend(loc=1)
    return plt

def plot_P(plt=None, mr=None, mv=None, title=None, Qstd=None, R=None, lpf_tau=None, data_lag=0.15):
    steady_only = False
    N = len(mr.Von)
    total_time = mr.time[-1]
    sample_freq_hz = float(N) / total_time
    sample_time = 1. / sample_freq_hz
    sample_freq_rps = sample_freq_hz * 2. * np.pi
    nyquist_freq_rps = sample_freq_rps / 2.
    min_possible_lpf_tau = 0.07 / nyquist_freq_rps * 50.
    print(f" nyquist {nyquist_freq_rps} r/s, min possible tau {min_possible_lpf_tau} s")
    mr_lpf = LagTustin(dt=sample_time, tau=data_lag, max_=3.3, min_=-3.3)

    # Get initial steady offset so can search for start of sweep.  Assume initial 50 seconds are steady.
    vec_initial = np.where( (mr.time <= 50.) & (mr.time >= 10.) )
    mr.Von = mr.Von - np.average(mr.Von[vec_initial])
    #plt.figure();     plq(plt, mr, 'time', mr, 'Von', color='blue', linestyle='-', label='Von')

    # Filter signal for cleaner statistical testing
    mr.Von_lpf = []
    mv.dt = mr.dt
    for i in range(N):
        mr.Von_lpf.append(mr_lpf.calculate(mr.Von[i], reset=i<1, dt=mr.dt[i]))
    mr.Von_lpf = np.array(mr.Von_lpf)
    steady_level_lpf = np.average(mr.Von_lpf[vec_initial])
    std_dev_lpf = np.std(mr.Von_lpf[vec_initial])
    mr.Von_lpf = mr.Von_lpf - steady_level_lpf

    try:
        index_start_sweep_lpf = np.array(np.where( abs(mr.Von_lpf) > 6.*std_dev_lpf))[0, 0]
        time_start_sweep_lpf = mr.time[index_start_sweep_lpf]
        index_end_sweep_lpf = np.where(mr.time < time_start_sweep_lpf + 150.)[0][-1]
        time_end_sweep_lpf = mr.time[index_end_sweep_lpf]
        print(f"{steady_level_lpf=} {std_dev_lpf=}")
        print(f"{index_start_sweep_lpf=} {time_start_sweep_lpf=}")
        print(f"{index_end_sweep_lpf=} {time_end_sweep_lpf=}")
    except IndexError:
        steady_only = True

    # Recenter mr.Von for freq analysis:  assume at least 179 sec fr
    # vec_fr = np.arange(index_start_sweep_lpf, index_end_sweep_lpf)
    vec_fr_for_avg = np.arange(index_start_sweep_lpf + int(0.5*(index_end_sweep_lpf - index_start_sweep_lpf)),
                               index_end_sweep_lpf)
    mr.Von = mr.Von - np.average(mr.Von[vec_fr_for_avg])
    mv.Von_kf = np.array(mv.Von_kf)
    mv.Von_kf = mv.Von_kf - np.average(mv.Von_kf[vec_fr_for_avg])
    mr.Von_lpf = mr.Von_lpf - np.average(mr.Von_lpf[vec_fr_for_avg])
    mv.Von_kf_avg = np.full((len(mv.Von_kf),), np.average(mv.Von_kf[vec_fr_for_avg]))

    plt.figure()
    plq(plt, mr, 'time', mr, 'Von', color='blue', linestyle='-', label='Von')
    plq(plt, mr, 'time', mr, 'Von_lpf', color='red', linestyle='--', label='Von_lpf')
    plq(plt, mv, 'time', mv, 'Von_kf', color='cyan', linestyle='-.', label='Von_kf')
    plq(plt, mv, 'time', mv, 'Von_kf_avg', color='orange', linestyle='-', label='Von_kf_avg')
    plt.text(0.5, 0.2, f"{Qstd=}   {R=} {lpf_tau=}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=12,
             color='blue',
             bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
    plt.legend(loc=1)
    plt.show(block=False)



    # steady_only = True

    if not steady_only:
        # Detect positive zero crossings
        is_positive = mr.Von_lpf[index_start_sweep_lpf:index_end_sweep_lpf]  > 0
        positive_crossings = (~is_positive[:-1]) & is_positive[1:]
        mr.crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep_lpf  # Add 1 to account for the shift
        mr.time_zero_crossing = mr.time[mr.crossing_indices]

        mv.time = mr.time
        is_positive = mv.Von_kf[index_start_sweep_lpf:index_end_sweep_lpf]  > 0
        positive_crossings = (~is_positive[:-1]) & is_positive[1:]
        mv.crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep_lpf  # Add 1 to account for the shift
        mv.time_zero_crossing = mv.time[mv.crossing_indices]

        # For simplicity assume mv zero crossing is always after mr zero crossing (lags behave like lags)
        # This also implies minimum phase behavior (magnitude decreasing) so normalize to max
        print("Time:    Frequency, Hz  /  Magnitude, dB   /  Phase, deg / raw_lag(s) / data_lag_lag(deg) / data_lag_lag(s) / lag(s)")
        transfer_function = []
        mag_normal = 0.
        for j in range(len(mr.time_zero_crossing)-1):
            time = mr.time_zero_crossing[j]
            index = mr.crossing_indices[j]
            period = mr.time_zero_crossing[j+1] - mr.time_zero_crossing[j]
            frequency = 1. / period
            ang_freq = frequency * 2. * np.pi
            data_lag_lag_deg = np.atan2(data_lag*ang_freq, 1.) * 180./np.pi
            data_lag_lag = data_lag_lag_deg / 360. * period
            raw_lag = mv.time_zero_crossing[j] - mr.time_zero_crossing[j]
            lag = raw_lag + data_lag_lag
            input = mr.Von_lpf[mr.crossing_indices[j]:mr.crossing_indices[j+1]]
            input_magnitude = max(input) - min(input)
            if j >= len(mv.crossing_indices) - 1:
                break
            response = mv.Von_kf[mv.crossing_indices[j]:mv.crossing_indices[j+1]]
            response_magnitude = (max(response)[0] - min(response)[0])
            tf_magnitude = 20.*np.log10(response_magnitude/input_magnitude)
            tf_phase = -360. * lag / period
            if frequency < 2.5:
                mag_normal = max(mag_normal, tf_magnitude)
            transfer_function.append([frequency, tf_magnitude, tf_phase, time, raw_lag, data_lag_lag_deg,
                                      data_lag_lag, lag, index])
            # print(f"{frequency}  /  {tf_magnitude}    / {tf_phase}")
        transfer_function = np.array(transfer_function)
        transfer_function[:, 1] -= mag_normal

        # Cleanup the result
        d = np.diff(transfer_function, axis=0)[:, 0]
        pos_indeces = np.where(d > 0.)
        transfer_function = transfer_function[pos_indeces]
        # Go through one-by-one and delete bad steps
        tf_clean = []
        tf_clean.append(transfer_function[0, :])
        k_clean = 0
        n = len(transfer_function[:, 0])
        k = 1
        while k < n:
            if (abs(transfer_function[k, 0] - tf_clean[k_clean][0]) < 1. and
                    (transfer_function[k, 0] - tf_clean[k_clean][0]) > 0.):
                tf_clean.append(transfer_function[k, :])
                k_clean += 1
            k += 1
        tf_clean = np.array(tf_clean)
        for j in range(len(tf_clean)):
            print("{:8.2f}: ".format(tf_clean[j][3]),
                  "{:7.1f} Hz / ".format(tf_clean[j][0]),
                  "{:7.1f} dB / ".format(tf_clean[j][1]),
                  "{:7.1f} deg / ".format(tf_clean[j][2]),
                  "{:7.3f} s  / ".format(tf_clean[j][4]),
                  "{:7.1f} deg  / ".format(tf_clean[j][5]),
                  "{:7.3f} s  / ".format(tf_clean[j][6]),
                  "{:7.3f} s  / ".format(tf_clean[j][7]),
                  )
        mv.time_clean = tf_clean[:, 3]
        mv.f_clean = tf_clean[:, 0]
        mv.w_clean = mv.f_clean * 2. * np.pi
        mv.mdB_clean = tf_clean[:, 1]
        mv.phs_clean = tf_clean[:, 2]

    # Metrics
    mr.Von_steady = mr.Von[vec_initial]
    mv.Von_kf_steady = mv.Von_kf[vec_initial]
    mr.Von_steady_lpf = mr.Von_lpf[vec_initial]
    mr.amp_Von_steady = np.max(mr.Von_steady) - np.min(mr.Von_steady)
    mv.amp_Von_kf_steady = np.max(mv.Von_kf_steady) - np.min(mv.Von_kf_steady)
    print(f" amp Von_kf_steady  {mv.amp_Von_kf_steady}   amp Von_steady {mr.amp_Von_steady}" )
    attenuation = mv.amp_Von_kf_steady / mr.amp_Von_steady
    attenuation_lpf = (np.max(mr.Von_steady_lpf) - np.min(mr.Von_steady_lpf)) / (np.max(mr.Von_steady) - np.min(mr.Von_steady))

    if not steady_only:
        phase45_tf_index = 0
        phase90_tf_index = 0
        db3_tf_index = 0
        for j in range(len(tf_clean)-1):
            frequency = tf_clean[j][0]
            phase_dg = tf_clean[j][2]
            mag_db = tf_clean[j][1]
            time = tf_clean[j][3]
            if mag_db < tf_clean[db3_tf_index][1] and mag_db >= -3.:
                db3_tf_index = j
            if  phase_dg < tf_clean[phase45_tf_index][2] and phase_dg >= -45.:
                phase45_tf_index =j
            if phase_dg < tf_clean[phase90_tf_index][2] and phase_dg >= -90.:
                phase90_tf_index = j
        freq_3db = tf_clean[db3_tf_index][0]
        tau_3db = 1. / (freq_3db * 2. * np.pi)
        mag_3db = tf_clean[db3_tf_index][1]
        time_3db = tf_clean[db3_tf_index][3]
        freq_45 = tf_clean[phase45_tf_index][0]
        tau_45 = 1. / (freq_45 * 2. * np.pi)
        phase_45 = tf_clean[phase45_tf_index][2]
        time_45 = tf_clean[phase45_tf_index][3]

        freq_90 = tf_clean[phase90_tf_index][0]
        omega_90 = freq_90 * 2. * np.pi
        tau_90 = 1. / (freq_90 * 2. * np.pi)
        phase_90 = tf_clean[phase90_tf_index][2]
        time_90 = tf_clean[phase90_tf_index][3]


        print(f"{attenuation=} {attenuation_lpf=}")
        print(f"{time_3db=} {freq_3db=} {mag_3db=}")
        print(f"{time_45=}  {freq_45=} {phase_45=}")
        print(f"{time_90=}  {freq_90=} {phase_90=} {omega_90=}")
    metric_string = "Metrics:\n"
    metric_string += "  Qstd = {:9.6f}\n  R =     {:9.6f}\n  lpf_tau = {:7.4f}\n\n".format(Qstd, R, lpf_tau)
    metric_string += "  Attn = {:5.2f}  Attn_lpf = {:5.2f}\n\n".format(attenuation, attenuation_lpf)
    if not steady_only:
        metric_string += "  -3db @    {:4.2f} Hz,  ({:5.1f} sec)\n".format(freq_3db, time_3db)
        metric_string += "  -45 deg @ {:4.2f} Hz   ({:5.1f} sec)\n\n".format(freq_45, time_45)
        metric_string += "  -90 deg @ {:4.2f} Hz   ({:5.1f} sec)\n\n".format(freq_90, time_90)
        metric_string += "  tau @ -3db = {:5.3f}\n  tau @ -45 = {:5.3f}\n  omega90 = {:5.3f}\n".format(tau_3db, tau_45, omega_90)
        res_title = "Qstd, R, lpf_tau, attenuation_lpf, amp_steady_kf, amp_steady, attenuation, tau_3db, tau_45, omega_90,"
        res = [Qstd, R, lpf_tau, attenuation_lpf, mv.amp_Von_kf_steady, mr.amp_Von_steady, attenuation, tau_3db, tau_45, omega_90]
    else:
        res_title = "Qstd, R, lpf_tau, attenuation_lpf,  amp_steady_kf, amp_steady, attenuation, tau_3db, tau_45, omega_90,"
        res = [Qstd, R, lpf_tau, attenuation_lpf, mv.amp_Von_kf_steady, mr.amp_Von_steady, attenuation,  0., 0., 0.]

    plt.figure()
    plt.figtext(0.1, 0.3, metric_string, fontsize=10, color='black', horizontalalignment='left',
                verticalalignment='center', bbox=dict(facecolor='orange', alpha=0.5, pad=5))
    plt.subplot(311)
    plt.title(title)
    plq(plt, mr, 'time', mr, 'Von', color='blue', linestyle='-', label='Von' + run_str)
    plq(plt, mr, 'time', mr, 'Von_lpf', color='cyan', linestyle='--', label='Von_lpf' + run_str)
    plq(plt, mv, 'time', mv, 'Von_kf', color='red', linestyle='--', label='Von_kf' + ver_str)
    plt.text(0.5, 0.2, f"{Qstd=}   {R=} {lpf_tau=}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=12,
             color='blue',
             bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
    plt.legend(loc=1)
    left_limit, right_limit = plt.xlim()
    plt.subplot(324)
    plt.semilogx(mv.w_clean, mv.mdB_clean, color='red', linestyle='-', label='mag_dB' + ver_str)
    plt.ylim([-18, 6])
    plt.legend(loc=1)
    plt.subplot(326)
    plt.semilogx(mv.w_clean, mv.phs_clean, color='red', linestyle='-', label='phs_deg' + ver_str)
    plt.ylim([-180, 0])
    plt.legend(loc=1)

    return plt, res, res_title

def running_rms(signal, window_size):
    """
    Calculates the running RMS amplitude of a signal using a sliding window.
    Args:
        signal (np.ndarray): The input signal (1D NumPy array).
        window_size (int): The size of the sliding window.
    Returns:
        np.ndarray: A NumPy array containing the running RMS values.
    """
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("Input signal must be a 1D NumPy array.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    if window_size > len(signal):
        raise ValueError("Window size cannot be greater than the signal length.")

    # Square the signal
    squared_signal = np.power(signal, 2)

    # Create a window of ones for averaging
    window = np.ones(window_size) / float(window_size)

    # Convolve the squared signal with the window to get the moving average of squares
    moving_average_of_squares = np.convolve(squared_signal, window, mode='valid')

    # Take the square root to get the running RMS
    running_rms_amplitude = np.sqrt(moving_average_of_squares)

    # Copy first window points to beginning to get same array size out as in
    running_rms_amplitude = np.insert(running_rms_amplitude, 0, running_rms_amplitude[0:window_size-1])

    return running_rms_amplitude


class Saved:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self):
        self.time = []
        self.dt = []
        self.pos = []
        self.velo = []


class KF1x1VarDt:
    """1x1 General Purpose Extended Kalman Filter.   Inherit from this class and include kf_predict and
    kf_update methods in the parent."""

    def __init__(self, initial_position, initial_velocity, dt, proc_noise_std, meas_noise_std):
        """
        Initializes a 1D Kalman filter with a constant velocity model.

        Args:
            initial_position (float): Initial estimate of the position.
            initial_velocity (float): Initial estimate of the velocity.
            dt (float): Time step between measurements.
            proc_noise_std (float): Standard deviation of the process noise (acceleration).
            meas_noise_std (float): Standard deviation of the measurement noise (position).
        """
        self.dt = dt
        self.Fx = np.array([[1.0, self.dt], [0.0, 1.0]])  # State transition
        self.G = None
        self.Bu = 0.  # Control transition
        # Process noise covariance matrix (assuming noise in acceleration)
        self.Q_std = proc_noise_std
        self.Q = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.R = np.array([meas_noise_std**2])  # State uncertainty.  Measurement noise covariance matrix
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]]) * 100  # Uncertainty covariance.  Large initial
        self.H = np.array([[1.0, 0.0]])  # Jacobian of h(x).  Measurement matrix (Only measure position)
        self.S = 0.  # System uncertainty
        self.K = 0.  # Kalman gain
        self.hx = 0.  # Output of observation function h(x)
        self.u_kf = 0.  # Control input
        self.x = np.array([[initial_position], [initial_velocity]])  # Kalman state vector [position, velocity]
        self.y_kf = 0.  # Residual z-hx
        self.y_kf_f = 0.  # Residual filtered z-hx
        self.z_kf = 0.  # Observation of state x
        self.x_prior = self.x
        self.P_prior = self.P
        self.x_post = self.x
        self.P_post = self.P
        self.tb_f_for_hx = 25.
        self.x_for_hx = 1.

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix + "KF1x1VarDt:\n"
        s += "  Inputs:\n"
        s += "  z = {:10.6g}\n".format(self.z_kf)
        s += "  Fx = \n" + self.Fx.__str__() + "\n"
        s += "  Bu = {:13.10g}\n".format(self.Bu)
        s += "  R = {:10.6g}\n".format(self.R[0])
        s += "  Q_std = {:10.6g}\n".format(self.Q_std)
        s += "  H = " + self.H.__str__() + "\n"
        s += "  Outputs:\n"
        s += "  x  = \n" + self.x.__str__() + "\n"
        s += "  hx = {:10.6g}\n".format(self.hx)
        s += "  y  = {:10.6g}\n".format(self.y_kf[0, 0])
        s += "  P  = \n" + self.P.__str__() + "\n"
        s += "  K  = \n" + self.K.__str__() + "\n"
        s += "  S  = {:10.6g}\n".format(self.S[0, 0])
        return s

    def predict(self, dt):
        """
        Performs the prediction step of the Kalman filter.
        Inputs:
            u   1x1 input, =ib, A
            Bu  1x1 control transition, Ohms
            Fx  2x2 state transition, V/V
        Outputs:
            x   2x1 Kalman state variable =
            P   2x2 Kalman probability
        """
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.Fx = np.array([[1.0, self.dt], [0.0, 1.0]])

        # Process noise covariance matrix (assuming noise affects acceleration)
        self.G = np.array([[0.5 * self.dt ** 2], [dt]])
        self.Q = self.G @ self.G.T * self.Q_std**2

        # Predict state and covariance
        self.x = self.Fx @ self.x
        self.P = self.Fx @ self.P @ self.Fx.T + self.Q

    def update(self, measurement):
        """
        Performs the update step of the Kalman filter.
        Updates the state estimate and covariance matrix based on the new measurement.

        Args:
            measurement (float): The new position measurement.

        Inputs:
            u   1x1 input, =ib, A
            Bu  1x1 control transition, Ohms
            Fx  2x2 state transition, V/V
        Outputs:
            S   1x1 Kalman gain
            K   2x1 Kalman gain
            H   1x2 Jacobian
            x   2x1 Kalman state variable = [input units, rate of change of input units]
            y_kf    1x1 output, units of input u (unity gain filter)
            P   2x2 Kalman probability matrix

        """
        # Kalman Gain
        self.S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

        # Update state estimate
        self.y_kf = measurement - (self.H @ self.x)  # Innovation
        self.x = self.x + (self.K @ self.y_kf)

        # Update covariance matrix

        self.P = (np.eye(self.x.shape[0]) - self.K @ self.H) @ self.P

    def init_kf(self, soc, p_init):
        """Initialize on demand"""
        self.x = soc
        self.P = p_init

    def h_jacobian(self, x):
        # implemented by child
        raise NotImplementedError

    def hx_calc(self):
        # implemented by child
        raise NotImplementedError

    def get_state(self):
        """
        Returns the current estimated state.

        Returns:
            numpy.ndarray: The current state vector [position, velocity].
        """
        return self.x

    def get_covariance(self):
        """
        Returns the current estimated covariance matrix.

        Returns:
            numpy.ndarray: The current covariance matrix.
        """
        return self.P


class KF1x1VarDt:
    """Explicit 1x1 General Purpose Extended Kalman Filter.   Inherit from this class and include kf_predict and
    kf_update methods in the parent."""

    def __init__(self, initial_position, initial_velocity, dt, proc_noise_std, meas_noise_std):
        """
        Initializes a 1D Kalman filter with a constant velocity model.

        Args:
            initial_position (float): Initial estimate of the position.
            initial_velocity (float): Initial estimate of the velocity.
            dt (float): Time step between measurements.
            proc_noise_std (float): Standard deviation of the process noise (acceleration).
            meas_noise_std (float): Standard deviation of the measurement noise (position).
        """
        self.dt = dt
        self.Fx = np.array([[1.0, self.dt], [0.0, 1.0]])  # State transition
        self.G  = None
        self.Bu = 0.  # Control transition
        # Process noise covariance matrix (assuming noise in acceleration)
        self.Q_std = proc_noise_std
        self.Q = self.P = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.R = meas_noise_std**2  # State uncertainty.  Measurement noise covariance matrix
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]]) * 100  # Uncertainty covariance.  Large initial
        self.H = np.array([[1.0, 0.0]])  # Jacobian of h(x).  Measurement matrix (Only measure position)
        self.S = 0.  # System uncertainty
        self.K = 0.  # Kalman gain
        self.hx = 0.  # Output of observation function h(x)
        self.u_kf = 0.  # Control input
        self.x = np.array([initial_position, initial_velocity])  # Kalman state vector [position, velocity]
        self.y_kf = 0.  # Residual z-hx
        self.y_kf_f = 0.  # Residual filtered z-hx
        self.z_kf = 0.  # Observation of state x
        self.x_prior = self.x
        self.P_prior = self.P
        self.x_post = self.x
        self.P_post = self.P
        self.tb_f_for_hx = 25.
        self.x_for_hx = 1.
        self.x = np.array([[0], [0]])

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix + "KF1x1VarDt:\n"
        s += "  Inputs:\n"
        s += "  z = {:10.6g}\n".format(self.z_kf)
        s += "  Fx = \n" + self.Fx.__str__() + "\n"
        s += "  Bu = {:13.10g}\n".format(self.Bu)
        s += "  R = {:10.6g}\n".format(self.R)
        s += "  Q_std = {:10.6g}\n".format(self.Q_std)
        s += "  H = " + self.H.__str__() + "\n"
        s += "  Outputs:\n"
        s += "  x  = \n" + self.x.__str__() + "\n"
        s += "  hx = {:10.6g}\n".format(self.hx)
        s += "  y  = {:10.6g}\n".format(self.y_kf)
        s += "  P  = \n" + self.P.__str__() + "\n"
        s += "  K  = \n" + self.K.__str__() + "\n"
        s += "  S  = {:10.6g}\n".format(self.S)
        return s

    def predict(self, dt):
        """
        Performs the prediction step of the Kalman filter.
        Inputs:
            u   1x1 input, =ib, A
            Bu  1x1 control transition, Ohms
            Fx  2x2 state transition, V/V
        Outputs:
            x   2x1 Kalman state variable =
            P   2x2 Kalman probability
        """
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.Fx = np.array([[1.0, self.dt], [0.0, 1.0]])
        Fx = self.Fx

        # Process noise covariance matrix (assuming noise affects acceleration)
        self.G = np.array([[0.5 * self.dt ** 2], [dt]])
        # self.Q = self.G @ G.T * self.Q_std**2
        self.Q = np.array([ [dt*dt/4, dt/2], [dt/2, 1]])*dt*dt*self.Q_std**2

        # Predict state and covariance
        # self.x = self.Fx @ self.x
        x = self.x
        self.x = np.array( [ [float(Fx[0, 0])*float(x[0, 0]) + float(Fx[0, 1])*float(x[1, 0])], [float(Fx[1, 0])*float(x[0, 0]) + float(Fx[1, 1])*float(x[1, 0])] ])
        p00 = self.P[0, 0]
        p01 = self.P[0, 1]
        p10 = self.P[1, 0]
        p11 = self.P[1, 1]
        q00 = self.Q[0, 0]
        q01 = self.Q[0, 1]
        q10 = self.Q[1, 0]
        q11 = self.Q[1, 1]
        # self.P = self.Fx @ self.P @ self.Fx.T + self.Q
        self.P = np.array(( [ [p00+p01*dt+p10*dt+p11*dt*dt + q00,  p01+p11*dt + q01], [p10+p11*dt + q10, p11 + q11] ] ))

    def update(self, measurement):
        """
        Performs the update step of the Kalman filter.
        Updates the state estimate and covariance matrix based on the new measurement.

        Args:
            measurement (float): The new position measurement.

        Inputs:
            u   1x1 input, =ib, A
            Bu  1x1 control transition, Ohms
            Fx  2x2 state transition, V/V
        Outputs:
            S   1x1 Kalman gain
            K   2x1 Kalman gain
            H   1x2 Jacobian
            x   2x1 Kalman state variable = [input units, rate of change of input units]
            y_kf    1x1 output, units of input u (unity gain filter)
            P   2x2 Kalman probability matrix

        """
        # Kalman Gain
        p00 = self.P[0, 0]
        p01 = self.P[0, 1]
        p10 = self.P[1, 0]
        p11 = self.P[1, 1]
        # self.S = self.H @ self.P @ self.H.T + self.R
        self.S = p00+self.R
        PHT = np.array([ [p00], [p10] ])
        # self.K = self.P @ self.H.T @ np.linalg.inv(self.S)
        self.K = np.array([ [p00], [p10] ]) * 1./(p00 + self.R)
        k0 = float(self.K[0,0])
        k1 = float(self.K[1,0])

        # Update state estimate
        # self.y_kf = measurement - (self.H @ self.x)  # Innovation
        self.y_kf = measurement - float(self.x[0,0])
        # self.x = self.x + (self.K @ self.y_kf)
        self.x = np.array( [ [float(self.x[0,0])+self.y_kf*self.K[0,0]], [float(self.x[1,0])+self.y_kf*self.K[1,0]] ] )

        # Update covariance matrix
        # self.P = (np.eye(self.x.shape[0]) - self.K @ self.H) @ self.P
        self.P = np.array( [[(1-k0)*p00, (1-k0)*p01], [-k1*p00+p10, -k1*p01+p11]])

    def init_kf(self, soc, p_init):
        """Initialize on demand"""
        self.x = soc
        self.P = p_init

    def h_jacobian(self, x):
        # implemented by child
        raise NotImplementedError

    def hx_calc(self):
        # implemented by child
        raise NotImplementedError

    def get_state(self):
        """
        Returns the current estimated state.

        Returns:
            numpy.ndarray: The current state vector [position, velocity].
        """
        return self.x

    def get_covariance(self):
        """
        Returns the current estimated covariance matrix.

        Returns:
            numpy.ndarray: The current covariance matrix.
        """
        return self.P


class SavedData:
    def __init__(self, x=None, time_end=None):

        if x is None:
            unit_x = None
            self.skip_x = None
            self.i = 0
            self.time = None
            self.dt = None
            self.Vcm = None
            self.Vom = None
            self.VoVcm = None
            self.Vcn = None
            self.Von = None
            self.VoVcn = None
            self.Tbv = None
            self.Vbv = None
        else:
            self.assign_all_from(x)

            # Special handling
            self.skip_x = np.bool(np.array(x.skip))
            self.i = 0
            self.time = np.array(x.time)
            self.dt = []
            for i in range(len(self.time)):
                if i == 0:
                    self.dt.append(self.time[1] - self.time[0])
                else:
                    self.dt.append(self.time[i] - self.time[i-1])

            # Truncate
            if time_end is not None:
                i_end = np.where(self.time <= time_end)[0][-1] + 1
                self.truncate(i_end, 'time')

    def __str__(self):
        s = "{},".format(self.unit[self.i])
        s += "{:8.6f},".format(self.Vca[self.i])
        s += "{:8.6f},".format(self.Voa[self.i])
        s += "{:8.6f},".format(self.VoVca[self.i])
        s += "{:8.6f},".format(self.Vcn[self.i])
        s += "{:8.6f},".format(self.Von[self.i])
        s += "{:8.6f},".format(self.VoVcn[self.i])
        s += "{:8.6f},".format(self.Tbv[self.i])
        s += "{:8.6f},".format(self.Vbv[self.i])
        return s

    def assign_all_from(self, x=None):
        """
        Iterates over members of a dataset x, assigns values to numpy.ndarray members
        """
        for name in list(x.dtype.names):
            setattr(self, name, x[name])

    def truncate(self, i_end=None, key_attr='time'):
        """
        Iterates over members of an self, assigns values to numpy.ndarray members
        from rap_self.ib up to i_end.
        """
        for attr_name in dir(self):
            # Filter out built-in attributes and methods
            if not attr_name.startswith('__') and not callable(getattr(self, attr_name)):
                member = getattr(self, attr_name)
                if isinstance(member, np.ndarray):
                    # Ensure the slice doesn't exceed the bounds of rap_self.ib
                    end_index = min(i_end, len(getattr(self, key_attr)))

                    # Assign the slice to the numpy.ndarray member
                    # If the target array has a different shape, direct assignment
                    # might fail or reshape the array. Using np.array() ensures
                    # a new array is created with the correct slice.
                    setattr(self, attr_name, getattr(self, attr_name)[:end_index])


# Load from files
def load_data(path_to_data, time_end_in):

    print(f"load_data: \n{path_to_data=}\n{time_end_in=}\n")

    hdr_key_x = "unit_x,"  # Find one self of title
    unit_key_x = "x_unit"

    data_file_clean = write_clean_file(path_to_data, type_='_x', hdr_key=hdr_key_x, unit_key=unit_key_x)
    if data_file_clean is None:
        return None, None, None, None, None, None
    if data_file_clean is not None:
        mon_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        mon_raw = None
        print(f"load_data: returning mon=None")

    mon = SavedData(x=mon_raw, time_end=time_end_in)

    return mon, data_file_clean


class Saved:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self):
        self.time = []
        self.dt = []
        self.Vca = []
        self.Voa = []
        self.VoVca = []
        self.Vcn = []
        self.Von = []
        self.VoVcn = []
        self.VoVcnA = []
        self.Tbv = []
        self.Vbv = []
        self.Von_kf = []


# Example Usage:
if __name__ == "__main__":
    """
    Test setup:  FY6900 Dominty Function Generator.  FY6900 CH 1 connected across shunt leads.
    (**** not this CH 2 ground connected to board ground.)
    Top level - Sweep.   - Freq 0.5 - 5.0, Ampl 0.01 - 0.01, Offs -0.01 - -0.01, Duty 50% - 50%,
    Mode Linear.   Direction Forth, Time 120s.  Turn off generator.
    'Cx27000',  wait 60 sec. Turn on generator and press OK on function generator.  When it reaches 0.5 Hz again press
    OK to stop.  Then turn off generator.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from DataOverModel import plq
    plt.rcParams['axes.grid'] = True

    time_end = None
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burstForKF_soc2p2_hi_lo_chg.csv'
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst470uF_ForKF_soc2p2_hi_lo_chg.csv'
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst220uF_ForKF_soc2p2_hi_lo_chg.csv'
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst100uF_ForKF_soc2p2_hi_lo_chg.csv'
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst47uF_ForKF_soc2p2_hi_lo_chg.csv'
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst0uF_VoVc_ForKF_soc2p2_hi_lo_chg.csv'
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burst100uF_VoVc_ForKF_soc2p2_hi_lo_chg.csv'


    """
    # Reconstruct and look at 2 vs 1 filter in Von
    Test setup:  FY6900 Dominty Function Generator.  FY6900 CH 1 connected across shunt leads.
    (**** not this CH 2 ground connected to board ground.)
    Top level - Sweep.   - Freq 0.5 - 5.0, Ampl 0.01 - 0.01, Offs 0.0 - 0.0 (to center Vo/Vc, Duty 50% - 50%,
    Mode Linear.   Direction Forth, Time 120s.  Turn off generator with.
    'Cx16000',  wait 60 sec. Turn on generator and press OK on function generator.  When it reaches 0.5 Hz again press
    OK to stop.  Then turn off generator.
    """
    data_file = './noise_study/burstForKF_Vo_Vc_Base.csv'  # Cx20000, Base
    # data_file = './noise_study/burstForKF_Vo_Vc_Gnd.csv'  # Cx20000, Pulldown to function generator ground using CH 2 probe
    # data_file = './noise_study/burstForKF_Vo_Vc_noPS.csv'  # Cx20000, Pulldown to function generator ground using CH 2 probe
    # data_file = './noise_study/burstForKF_Vo_Vc_noBT.csv'  # Cx20000, Pulldown to function generator ground using CH 2 probe

    mr, data_file_clean = load_data(data_file, time_end)
    title = 'Vo Base kfDemo.py var dt'
    dt = 0.1  # Time step (seconds) used only on init

    # The best design of filter
    Qstd = 0.015  # Standard deviation of acceleration noise
    lpf_tau = 0.008
    R = 0.001  # Standard deviation of voltage measurement noise

    have_Vcm = False
    have_Vom = False
    have_Vcn = False
    have_Von = False
    have_VoVcm = False
    have_VoVcn = False
    have_Tbv = False
    have_Vbv = False

    if hasattr(mr, 'Vca'):
        have_Vca = True
        kfVca = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'Voa'):
        have_Voa = True
        kfVoa = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'Vcn'):
        have_Vcn = True
        kfVcn = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'Von'):
        have_Von = True
        kfVon = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'VoVca'):
        have_VoVca = True
        kfVoVca = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd*2., meas_noise_std=R*2.)
    if hasattr(mr, 'VoVcn'):
        have_VoVcn = True
        kfVoVcn = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd*2., meas_noise_std=R*2.)
    if hasattr(mr, 'Tbv'):
        have_Tbv = True
        kfTbv = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'Vbv'):
        have_Vbv = True
        kfVbv = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)

    run_str = '_burst data'
    ver_str = '_filtered'

    title = 'Part Factorial kfDemo.py var dt'

    # General purpose results
    # Data structures
    # if have_Vca and have_VoVca and have_Tbv and have_Vbv and have_VoVcn and have_Vcn and have_Von:
    if have_VoVcn and have_Vcn and have_Von:
        mv = Saved()
        v_rat = None

        for i in range(len(mr.time)):
            mv.time.append(mr.time[i])

            if hasattr(mr, 'Voa'):
                kfVoa.predict(mr.dt[i])
                kfVoa.update(mr.Voa[i])
                vf, v_rat = kfVoa.get_state()
                mv.Voa.append(vf[0])

            if hasattr(mr, 'Vca'):
                kfVca.predict(mr.dt[i])
                kfVca.update(mr.Vca[i])
                vf, v_rat = kfVca.get_state()
                mv.Vca.append(vf[0])

            if hasattr(mr, 'VoVca'):
                kfVoVca.predict(mr.dt[i])
                kfVoVca.update(mr.VoVca[i])
                vf, v_rat = kfVoVca.get_state()
                mv.VoVca.append(vf[0])

            if hasattr(mr, 'Von'):
                kfVon.predict(mr.dt[i])
                kfVon.update(mr.Von[i])
                vf, v_rat = kfVon.get_state()
                mv.Von.append(vf[0])

            if hasattr(mr, 'Vcn'):
                kfVcn.predict(mr.dt[i])
                kfVcn.update(mr.Vcn[i])
                vf, v_rat = kfVcn.get_state()
                mv.Vcn.append(vf[0])

            if hasattr(mr, 'VoVcn'):
                kfVoVcn.predict(mr.dt[i])
                kfVoVcn.update(mr.VoVcn[i])
                vf, v_rat = kfVoVcn.get_state()
                mv.VoVcn.append(vf[0])
                mv.VoVcnA.append(mv.Von[i] - mv.Vcn[i])

            if hasattr(mr, 'Vbv'):
                kfVbv.predict(mr.dt[i])
                kfVbv.update(mr.Vbv[i])
                vf, v_rat = kfVbv.get_state()
                mv.Vbv.append(vf[0])

            if hasattr(mr, 'Tbv'):
                kfTbv.predict(mr.dt[i])
                kfTbv.update(mr.Tbv[i])
                vf, v_rat = kfTbv.get_state()
                mv.Tbv.append(vf[0])


        plt = plot_1(plt, mr, mv, title+' F1')
        plt = plot_2(plt, mr, mv, title+' F2')
        plt = plot_3(plt, mr, mv, title+' F3')
        plt = plot_4(plt, mr, mv, title+' F4')
        plt = plot_5(plt, mr, mv, title+' F5')
        mv.VofVcfn = []
        for i in range(len(mv.time)):
            mv.VofVcfn.append(float(mv.Von[i] - mv.Vcn[i]))
        mv.VofVcfn = np.array(mv.VofVcfn)
        plt = plot_6(plt, mr, mv, title+' F5')


    doing_doe = True
    if doing_doe:
        Res = []
        # for Qstd, R, lpf_tau in \
        #         [
        #             [0.015, 0.001, 0.00001],  [0.03, 0.001, 0.00001],  [0.0075, 0.001, 0.00001],  [0.015, 0.002, 0.00001], [0.015, 0.0005, 0.00001],
        #             [0.015, 0.0005, 0.00001], [0.03, 0.0005, 0.00001], [0.0075, 0.0005, 0.00001], [0.015, 0.001, 0.00001], [0.015, 0.00025, 0.00001],
        #             [0.015, 0.001, 0.100],  [0.03, 0.001, 0.100],  [0.0075, 0.001, 0.100],  [0.015, 0.002, 0.100], [0.015, 0.0005, 0.100],
        #             [0.015, 0.0005, 0.100], [0.03, 0.0005, 0.100], [0.0075, 0.0005, 0.100], [0.015, 0.001, 0.100], [0.015, 0.00025, 0.100],
        #             [1.5, 0.00001, 0.00001], [1.5, 0.00001, 0.050], [1.5, 0.00001, 0.100], [1.5, 0.00001, 0.150], [1.5, 0.00001, 0.250],
        #           ]:
        for Qstd, R, lpf_tau in \
            [
                [0.015,  0.001,   0.00001], [0.03,   0.001,   0.00001], [0.0015, 0.001,   0.00001],  [0.015, 0.002, 0.00001],
                [0.015,  0.0001,  0.00001], [0.03,   0.0001,  0.00001], [0.0015, 0.0001,  0.00001],
                [0.015,  0.00001, 0.00001], [0.03,   0.00001, 0.00001], [0.0015, 0.00001, 0.00001],
                [0.0015, 0.001,   0.00001], [0.0015, 0.002,   0.00001], [0.0015, 0.0001,  0.00001],
                [1.5, 0.00001, 0.00001],
              ]:
        # for Qstd, R, lpf_tau in [
        #     [0.015, 0.001, 0.100], [0.015, 0.001, 0.008], [0.015, 0.001, 0.00001],
        #     [0.150, 0.0001, 0.100], [0.15, 0.0001, 0.008], [0.15, 0.001, 0.00001],
        #     ]:
        # for Qstd, R, lpf_tau in [ [0.015, 0.001, 0.100] ]:
            print(f"{Qstd=} {R=} {lpf_tau=}")
            kfVon = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                               proc_noise_std=Qstd, meas_noise_std=R)
            kfVonX = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                               proc_noise_std=Qstd, meas_noise_std=R)
            lpfVon = LagExp(dt=dt, tau=lpf_tau, min_=-3.3, max_=3.3)


            run_str = '_burst data'
            ver_str = '_filtered'

            # Data structures
            mv = Saved()
            v_rat = None

            for i in range(len(mr.time)):
                mv.time.append(mr.time[i])

                kfVon.predict(mr.dt[i])
                # kfVonX.predict(mr.dt[i])
                kfVon.update(mr.Von[i])
                # kfVonX.update(mr.Von[i])
                if i > 3:
                    pass
                Von_kf, v_rat = kfVon.get_state()
                mv.Von_kf.append(Von_kf)

                vf_lpf = lpfVon.calculate_tau(Von_kf[0], i<1, mr.dt[i], lpf_tau)
                mv.Von.append(vf_lpf)

            # plt.figure()
            # plt.subplot(111)
            # plt.title(title)
            # plq(plt, mr, 'time', mr, 'Von', color='blue', linestyle='-', label='Von' + run_str)
            # plq(plt, mr, 'time', mr, 'Von_lpf', color='cyan', linestyle='-', label='Von_lpf' + run_str)
            # plq(plt, mv, 'time', mv, 'Von_kf', color='red', linestyle='-.', label='Von_kf' + ver_str)
            # plq(plt, mv, 'time', mv, 'Von', color='magenta', linestyle=':', label='Von' + ver_str)
            # plt.legend(loc=1)
            # plt.show()
            #

            plt, res, res_title = plot_P(plt, mr, mv, title + ' P1', Qstd=Qstd, R=R, lpf_tau=lpf_tau)
            Res.append(res)

        # Summarize
        print(f"{res_title}")
        for i in range(len(Res)):
            print("{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},"\
                  .format(Res[i][0], Res[i][1], Res[i][2], Res[i][3], Res[i][4], Res[i][5], Res[i][6], Res[i][7], Res[i][8], Res[i][9]))
        csv_file = 'KF1x1.csv'
        with open(csv_file, "w") as output:
            output.write(res_title + '\n')
            for i in range(len(Res)):
                output.write("{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},{:9.6f},\n" \
                             .format(Res[i][0], Res[i][1], Res[i][2], Res[i][3], Res[i][4], Res[i][5], Res[i][6], Res[i][7], Res[i][8], Res[i][9]))




    plt.show(block=True)



