# BatteryKF - general purpose battery class for embedded KF
# Copyright (C) 2026 Dave Gutz
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
from filter.myFilters import General2Pole, LagExp
from filter.KF1x1 import KF1x1VarDt
from itertools import pairwise

def plot_1(plt=None, mr=None, mv=None, title=None):
    plt.figure()
    print("plot_11:", end='')
    plt.subplot(211)
    plt.title(title)
    plq(plt, mr, 'time', mr, 'VoVcn', color='blue', linestyle='-', label='VoVcn' + run_str)
    plq(plt, mr, 'time', mr, 'VoVcn_kf', color='red', linestyle='-', label='VoVcn_kf' + run_str)
    plq(plt, mr, 'time', mr, 'VoVcn_filt', color='black', linestyle='-', label='VoVcn_filt' + run_str)
    plq(plt, mr, 'time', mr, 'VoVcn_lag', color='cyan', linestyle='-.', label='VoVcn_lag' + run_str)
    top_limit, bottom_limit = plt.ylim()
    plt.legend(loc=1)
    plt.subplot(212)
    plq(plt, mr, 'time', mr, 'VoVcn_kf', color='red', linestyle='-', label='VoVcn_kf' + run_str)
    plq(plt, mr, 'time', mr, 'VoVcn_filt', color='black', linestyle='-', label='VoVcn_filt' + run_str)
    plt.legend(loc=1)
    plt.ylim(top_limit, bottom_limit)
    return plt

def plot_P(plt=None, mr=None, mv=None, title=None, Qstd=None, Rstd=None, data_lag=None):
    steady_only = False
    mv.dt = mr.dt

    # Get initial steady offset so can search for start of sweep.  Assume initial 50 seconds are steady.
    vec_initial = np.where( (mr.time <= 50.) & (mr.time >= 10.) )
    mr.VoVcn_avg = np.average(mr.VoVcn[vec_initial])
    mr.VoVcn = mr.VoVcn - mr.VoVcn_avg
    mr.VoVcn_kf = mr.VoVcn_kf - mr.VoVcn_avg
    mr.VoVcn_filt = mr.VoVcn_filt - mr.VoVcn_avg
    mr.VoVcn_lag = mr.VoVcn_lag - mr.VoVcn_avg
    std_dev_lag = np.std(mr.VoVcn_lag[vec_initial])

    try:
        index_start_sweep_lag = np.array(np.where( mr.VoVcn_lag < -10.*std_dev_lag))[0, 0]
        time_start_sweep_lag = mr.time[index_start_sweep_lag]
        index_end_sweep_lag = np.where(mr.time < time_start_sweep_lag + 650.)[0][-1]
        time_end_sweep_lag = mr.time[index_end_sweep_lag]
        print(f"{steady_level_lag=} {std_dev_lag=}")
        print(f"{index_start_sweep_lag=} {time_start_sweep_lag=}")
        print(f"{index_end_sweep_lag=} {time_end_sweep_lag=}")
    except IndexError:
        steady_only = True

    # Recenter mr.VoVcn for freq analysis:  assume at least 179 sec fr
    # vec_fr = np.arange(index_start_sweep_lag, index_end_sweep_lag)
    if not steady_only:
        vec_fr_for_avg = np.arange(index_start_sweep_lag + int(0.5*(index_end_sweep_lag - index_start_sweep_lag)),
                                   index_end_sweep_lag)
    else:
        vec_fr_for_avg = vec_initial

    # Center
    mr.VoVcn_avg = np.average(mr.VoVcn[vec_fr_for_avg])
    mr.VoVcn = mr.VoVcn - mr.VoVcn_avg
    mr.VoVcn_kf = np.array(mr.VoVcn_kf - mr.VoVcn_avg)
    mr.VoVcn_filt = np.array(mr.VoVcn_filt - mr.VoVcn_avg)
    mr.VoVcn_lag = np.array(mr.VoVcn_lag - mr.VoVcn_avg)

    mv.VoVcn_kf = np.array(mv.VoVcn_kf)
    mv.VoVcn_kf_avg = np.average(mv.VoVcn_kf[vec_fr_for_avg])
    mv.VoVcn_kf = mv.VoVcn_kf - mv.VoVcn_kf_avg
    mv.VoVcn_kf_avg = np.average(mv.VoVcn_kf[vec_fr_for_avg])  # recalculate

    mv.VoVcn_kf_avg = np.full((len(mv.VoVcn_kf),), mv.VoVcn_kf_avg)

    plt.figure()
    print("plot_12:", end='')
    plt.subplot(211)
    plt.title(title+'1')
    plq(plt, mr, 'time', mr, 'VoVcn', color='blue', linestyle='-', label='VoVcn burst_data centered')
    plq(plt, mr, 'time', mr, 'VoVcn_kf', color='red', linestyle='-', label='VoVcn_kf burst_data centered')
    plq(plt, mr, 'time', mr, 'VoVcn_filt', color='black', linestyle='-', label='VoVcn_filt burst_data centered')
    plq(plt, mr, 'time', mr, 'VoVcn_lag', color='cyan', linestyle='--', label='VoVcn_lag burst_data')
    plq(plt, mv, 'time', mv, 'VoVcn_kf', color='pink', linestyle='-.', label='VoVcn_kf calc')
    plq(plt, mv, 'time', mv, 'VoVcn_kf_avg', color='orange', linestyle='-', label='VoVcn_kf calc avg')
    plt.text(0.5, 0.2, f"{Qstd=} Rstd={Rstd}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=12,
             color='blue',
             bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
    top_limit, bottom_limit = plt.ylim()
    plt.legend(loc=1)
    plt.subplot(212)
    plq(plt, mr, 'time', mr, 'VoVcn_kf', color='red', linestyle='-', label='VoVcn_kf burst_data centered')
    plq(plt, mr, 'time', mr, 'VoVcn_filt', color='black', linestyle='-', label='VoVcn_filt burst_data centered')
    plq(plt, mv, 'time', mv, 'VoVcn_kf', color='pink', linestyle='-.', label='VoVcn_kf calc')
    plq(plt, mv, 'time', mv, 'VoVcn_kf_avg', color='orange', linestyle='-', label='VoVcn_kf calc avg')
    plt.ylim(top_limit, bottom_limit)
    plt.legend(loc=1)

    plt.show(block=False)

    # steady_only = True
    if not steady_only:
        # Detect positive zero crossings
        is_positive = mr.VoVcn_lag[index_start_sweep_lag:index_end_sweep_lag]  > 0
        positive_crossings = (~is_positive[:-1]) & is_positive[1:]
        mr.crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep_lag  # Add 1 to account for the shift
        mr.time_zero_crossing = mr.time[mr.crossing_indices]

        mv.time = mr.time
        is_positive = mv.VoVcn_kf[index_start_sweep_lag:index_end_sweep_lag]  > 0
        positive_crossings = (~is_positive[:-1]) & is_positive[1:]
        mv.crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep_lag  # Add 1 to account for the shift
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
            input = mr.VoVcn_lag[mr.crossing_indices[j]:mr.crossing_indices[j+1]]
            input_magnitude = max(input) - min(input)
            if j >= len(mv.crossing_indices) - 1:
                break
            response = mv.VoVcn_kf[mv.crossing_indices[j]:mv.crossing_indices[j+1]]
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
    mr.VoVcn_steady = mr.VoVcn[vec_initial]
    mv.VoVcn_kf_steady = mv.VoVcn_kf[vec_initial]
    mr.VoVcn_steady_lag = mr.VoVcn_lag[vec_initial]
    mr.amp_VoVcn_steady = np.max(mr.VoVcn_steady) - np.min(mr.VoVcn_steady)
    mv.amp_VoVcn_kf_steady = np.max(mv.VoVcn_kf_steady) - np.min(mv.VoVcn_kf_steady)
    mr.amp_VoVcn_steady_lag = np.max(mr.VoVcn_steady_lag) - np.min(mr.VoVcn_steady_lag)
    print(f" amp VoVcn_kf_steady  {mv.amp_VoVcn_kf_steady}   amp VoVcn_steady {mr.amp_VoVcn_steady}" )
    attenuation = mv.amp_VoVcn_kf_steady / mr.amp_VoVcn_steady
    attenuation_lag = mr.amp_VoVcn_steady_lag / mr.amp_VoVcn_steady

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


        print(f"{attenuation=} {attenuation_lag=}")
        print(f"{time_3db=} {freq_3db=} {mag_3db=}")
        print(f"{time_45=}  {freq_45=} {phase_45=}")
        print(f"{time_90=}  {freq_90=} {phase_90=} {omega_90=}")
    metric_string = "Metrics:\n"
    metric_string += "  Qstd = {:9.6f}\n  Rstd =     {:9.6f}\n  data_lag = {:7.4f}\n\n".format(Qstd, Rstd, data_lag)
    metric_string += "  Attn = {:5.2f}  Attn_lag = {:5.2f}\n\n".format(attenuation, attenuation_lag)
    if not steady_only:
        metric_string += "  -3db @    {:4.2f} Hz,  ({:5.1f} sec)\n".format(freq_3db, time_3db)
        metric_string += "  -45 deg @ {:4.2f} Hz   ({:5.1f} sec)\n\n".format(freq_45, time_45)
        metric_string += "  -90 deg @ {:4.2f} Hz   ({:5.1f} sec)\n\n".format(freq_90, time_90)
        metric_string += "  tau @ -3db = {:5.3f}\n  tau @ -45 = {:5.3f}\n  omega90 = {:5.3f}\n".format(tau_3db, tau_45, omega_90)
        res_title = "Qstd, Rstd, data_lag, attenuation_lag, amp_steady_kf, amp_steady, attenuation, tau_3db, tau_45, omega_90,"
        res = [Qstd, Rstd, data_lag, attenuation_lag, mv.amp_VoVcn_kf_steady, mr.amp_VoVcn_steady, attenuation, tau_3db, tau_45, omega_90]
    else:
        res_title = "Qstd, Rstd, data_lag, attenuation_lag,  amp_steady_kf, amp_steady, attenuation, tau_3db, tau_45, omega_90,"
        res = [Qstd, Rstd, data_lag, attenuation_lag, mv.amp_VoVcn_kf_steady, mr.amp_VoVcn_steady, attenuation,  0., 0., 0.]

    plt.figure()
    print("plot_P1:", end='')
    plt.figtext(0.1, 0.3, metric_string, fontsize=10, color='black', horizontalalignment='left',
                verticalalignment='center', bbox=dict(facecolor='orange', alpha=0.5, pad=5))
    plt.subplot(311)
    plt.title(title+'2')
    plq(plt, mr, 'time', mr, 'VoVcn', color='blue', linestyle='-', label='VoVcn' + run_str)
    plq(plt, mr, 'time', mr, 'VoVcn_kf', color='red', linestyle='-', label='VoVcn_kf' + run_str)
    plq(plt, mr, 'time', mr, 'VoVcn_filt', color='black', linestyle='-', label='VoVcn_filt' + run_str)
    plq(plt, mr, 'time', mr, 'VoVcn_lag', color='cyan', linestyle='--', label='VoVcn_lag' + run_str)
    plq(plt, mv, 'time', mv, 'VoVcn_kf', color='pink', linestyle='-.', label='VoVcn_kf' + ver_str)
    plt.text(0.5, 0.2, f"{Qstd=} Rstd={Rstd}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=12,
             color='blue',
             bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
    plt.legend(loc=1)
    left_limit, right_limit = plt.xlim()
    if not steady_only:
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
            self.VoVcn = None
            self.VoVcn = None
            self.Tbv = None
            self.Vbv = None
        else:
            self.assign_all_from(x)

            # Special handling
            self.skip_x = np.bool(np.array(x.skip_shunt))
            self.i = 0
            try:
                self.time = np.array(x.time)
            except AttributeError:
                self.time = np.array(x.c_time) - x.c_time[0]

            self.dt = [b - a for a, b in pairwise(self.time)]
            self.dt.insert(0, self.dt[0])

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
        s += "{:8.6f},".format(self.VoVcn[self.i])
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
def load_data_KF1x1_test(path_to_data, time_end):

    print(f"load_data_KF1x1_test: \n{path_to_data=}\n{time_end=}\n")

    hdr_key_x = "unit_shunt,"  # Find one self of title
    unit_key_x = "shunt_unit"

    data_file_clean = write_clean_file(path_to_data, type_='_shunt', hdr_key=hdr_key_x, unit_key=unit_key_x)
    if data_file_clean is None:
        return None, None, None, None, None, None
    import numpy as np
    if data_file_clean is not None:
        mon_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        mon_raw = None
        print(f"load_data_KF1x1_test: returning mon=None")

    mon = SavedData(x=mon_raw, time_end=time_end)

    return mon, data_file_clean


class Saved:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self):
        self.time = []
        self.dt = []
        self.VoVcn = []
        self.VoVcn_kf = []
        self.VoVcn_filt = []


# Example Usage:  ran ok 20260217
if __name__ == "__main__":
    """
    Test setup:  FY6900 Dominty Function Generator.  FY6900 CH 1 connected across shunt leads.
    (**** not this CH 2 ground connected to board ground.)
    VCO level Ampl .001 - 0, Offs 0-0, Freq 0-0, Duty 50-50
    Top level - Sweep.   - Freq 0 - 5.0, Ampl 0.01 - 0.01, Offs 0.0 - 0.0, Duty 50% - 50%,
                    Mode Log.   Direction Forth, Time 360s.  OK - OK quickly to freeze it at 0 Hz then quickly VCO - OK
    'clear' in GUI then 'Cx45000' on tty,  wait 60 sec. Turn on generator and press OK on function generator.  When it reaches 5.0 Hz again press
    OK to stop.  Then press VCO - OK quickly
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from plot.plq import plq as plq
    plt.rcParams['axes.grid'] = True
    plt.rcParams['legend.fontsize'] = 'small'
    from butterHighPassDemo import butter_highpass_filter
    time_end = None

    """
    # Reconstruct and look at 2 vs 1 filter in VoVcn
    Test setup:  FY6900 Dominty Function Generator.  FY6900 CH 1 connected across shunt leads.
    (**** not this CH 2 ground connected to board ground.)
    Top level - Sweep.   - Freq 0.5 - 5.0, Ampl 0.01 - 0.01, Offs 0.0 - 0.0 (to center Vo/Vc, Duty 50% - 50%,
    Mode Linear.   Direction Forth, Time 120s.  Turn off generator with.
    'Cx16000',  wait 60 sec. Turn on generator and press OK on function generator.  When it reaches 0.5 Hz again press
    OK to stop.  Then turn off generator.
    """

    # data_file = './noise_study/ssnoise_soc2p2_hi_lo_chg_shunt.csv'  # Cx20000, Base
    # data_file = './noise_study/sschirp_soc2p2_hi_lo_chg.csv'  # Cx4800, Base
    # data_file = './noise_study/sweepchirp_soc2p2_hi_lo_chg.csv'  # Cx46000, Base


    """
    # Reconstruct and look at 2 vs 1 filter in VoVcn
    0.  Test setup:  FY6900 Dominty Function Generator.  FY6900 CH 1 connected across shunt leads.
    (**** not this CH 2 ground connected to board ground.)
    Top level - Sweep.   - Freq 0.0 - 5.0, Ampl 0.015 - 0.05, Offs 0.0 - 0.0 (to center Vo/Vc, Duty 50% - 50%,
    Mode Linear.   Direction Forth, Time 720s.  
    1.  Prep:  VCO OK to turn off generator with.  Run a few Cx1000 runs to make sure vonkf is steady.  Clear on GUI
    2.  Press Cx16000 to collect ss data for 60s
    3.  After 60 s press Sweep then OK.  When it reaches 5.0 Hz again press OK to stop then VCO OK to go back steady
    """
    data_file = './noise_study/sweepchirp1_soc2p2_hi_lo_chg.csv'  # Cx46000, new base 20251231
    doing_doe = False  # Toggle this to see various kf implemented in python
    cutoff_freq_hz = 0.05  # hpf


    """
    # Reconstruct and look at 2 vs 1 filter in VoVcn
    0.  Test setup:  FY6900 Dominty Function Generator.  FY6900 CH 1 connected across shunt leads.
    (**** not this CH 2 ground connected to board ground.)
    Top level - Sweep.   - Freq 0.0 - 5.0, Ampl 0.1 - 1., Offs 0.0 - 0.0 (to center Vo/Vc, Duty 50% - 50%,
    Mode Linear.   Direction Forth, Time 720s.  
    1.  Prep:  VCO OK to turn off generator with.  Run a few Cx1000 runs to make sure vonkf is steady.  Clear on GUI
    2.  Press Cx16000 to collect ss data for 60s
    3.  After 60 s press Sweep then OK.  When it reaches 5.0 Hz again press OK to stop then VCO OK to go back steady
    """
    data_file = './noise_study/sweepchirp3_soc2p2_hi_lo_chg.csv'  # Cx46000, new base 20251231
    doing_doe = True  # Toggle this to see various kf implemented in python
    cutoff_freq_hz = 0.05  # hpf
    Qstd = 0.0003  # Standard deviation of acceleration noise
    Rstd = 0.1000  # Standard deviation of voltage measurement noise


    """
    # Reconstruct and look at 2 vs 1 filter in VoVcn
    0.  Test setup:  FY6900 Dominty Function Generator.  FY6900 CH 1 connected across shunt leads.
    (**** not this CH 2 ground connected to board ground.)
    Top level - Sweep.   - Freq 0.0 - 5.0, Ampl 0.1 - 1., Offs 0.0 - 0.0 (to center Vo/Vc, Duty 50% - 50%,
    Mode Linear.   Direction Forth, Time 720s.  
    1.  Prep:  VCO OK to turn off generator with.  Run a few Cx1000 runs to make sure vonkf is steady.  Clear on GUI
    2.  Press Cx16000 to collect ss data for 60s
    3.  After 60 s press Sweep then OK.  When it reaches 5.0 Hz again press OK to stop then VCO OK to go back steady
    """
    doing_doe = True  # Toggle this to see various kf implemented in python
    cutoff_freq_hz = 0.05  # hpf
    # The best design of filter
    Qstd = 0.0003  # Standard deviation of acceleration noise
    Rstd = 0.0100  # Standard deviation of voltage measurement noise



############################################################33
    mr, data_file_clean = load_data_KF1x1_test(data_file, time_end)
    title = 'VoVc Base KF1x1_test.py var dt'
    dt = 0.1  # Time step (seconds) used only on init

    # Some initializations
    # Utility measurement lag for finding zero crossings quietly
    data_lag = None

    # High pass on data
    # Filter signal for cleaner statistical testing
    N = len(mr.time)
    total_time = mr.time[-1]
    sample_freq_hz = float(N) / total_time
    sample_time = 1. / sample_freq_hz
    sample_freq_rps = sample_freq_hz * 2. * np.pi
    nyquist_freq_rps = sample_freq_rps / 2.
    min_possible_data_lag = 0.07 / nyquist_freq_rps * 50.
    vec_initial = np.where( (mr.time <= 50.) & (mr.time >= 10.) )
    data_lag = min_possible_data_lag * 5.
    print(f" nyquist {nyquist_freq_rps} r/s, min possible tau {min_possible_data_lag} s, data_lag {data_lag}, s")
    mr.VoVcn = butter_highpass_filter(mr.vovcn, cutoff_freq_hz, sample_freq_hz, 2)
    mr.VoVcn_kf = butter_highpass_filter(mr.vovcnkf, cutoff_freq_hz, sample_freq_hz, 2)
    mr_lag = LagExp(dt=sample_time, tau=data_lag, max_=3.3, min_=-3.3)
    mr.VoVcn_lag = []

    mr.VoVcn_lag = [mr_lag.calculate(mr.VoVcn[i], reset=i<1, dt=mr.dt[i]) for i in range(N)]


    mr.VoVcn_lag = np.array(mr.VoVcn_lag)
    steady_level_lag = np.average(mr.VoVcn_lag[vec_initial])
    mr.VoVcn_lag = np.array(mr.VoVcn_lag)

    # Old 2-pole for reference
    VoVcnFilt = General2Pole(0.1, 0.5, 0.8, -3., 3.)
    mr.VoVcn_filt = []
    for i in range(N):
        if i==0:
            lagged_val2 = VoVcnFilt.calculate(mr.VoVcn_kf[i], reset=True, dt=mr.dt[i])
        else:
            lagged_val2 = VoVcnFilt.calculate(mr.VoVcn[i], reset=False, dt=mr.dt[i])
        mr.VoVcn_filt.append(lagged_val2)
    mr.VoVcn_filt = np.array(mr.VoVcn_filt)

    # Local kf
    kfVoVcn = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt, proc_noise_std=Qstd*2.,
                         meas_noise_std=Rstd*2.)

    run_str = ' chirp data'
    ver_str = ' calc'

    mv = Saved()
    v_rat = None

    for i in range(len(mr.time)):
        mv.time.append(mr.time[i])

        kfVoVcn.predict(mr.dt[i])
        kfVoVcn.update(mr.VoVcn[i])
        vf, v_rat = kfVoVcn.get_state()
        mv.VoVcn.append(vf[0])

    plt = plot_1(plt, mr, mv, title + ' F1')

    if doing_doe:
        ii = 0
        Res = []
        for Qstd, Rstd in [ [0.0003, 0.100], [0.0003, 0.010], [0.003, 0.001] ]:
        # for Qstd, Rstd in \
        #         [
        #             [0.0003,  0.0100], [0.0003, 0.1000],
        #             [0.0006,  0.1000], [0.00015, 0.1000],
        #             [0.0003,  0.2000], [0.0003,  0.0500],
        #             [0.0006,  0.0100], [0.00015, 0.0100],
        #             [0.0003,  0.0100], [0.0006, 0.0100],  [0.00015, 0.0100],
        #             [0.00003, 0.0100], [0.00003, 0.0200], [0.00003, 0.0050],
        #             [1.5,    0.00001], [0.0003,  0.1000],
        #         ]:
            ii += 1

            print(f"{Qstd=} {Rstd=}")
            kfVoVcn = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                                 proc_noise_std=Qstd, meas_noise_std=Rstd)
            kfVoVcnX = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                                  proc_noise_std=Qstd, meas_noise_std=Rstd)
            lagVoVcn = LagExp(dt=dt, tau=data_lag, min_=-3.3, max_=3.3)


            run_str = '_chirp_data'
            ver_str = '_calc'

            # Data structures
            mv = Saved()
            v_rat = None

            for i in range(len(mr.time)):
                mv.time.append(mr.time[i])

                # kfVoVcnX.predict(mr.dt[i])
                # kfVoVcnX.update(mr.VoVcn[i])
                kfVoVcn.predict(mr.dt[i])
                kfVoVcn.update(mr.VoVcn[i])
                if i > 3:
                    pass
                VoVcn_kf, v_rat = kfVoVcn.get_state()
                mv.VoVcn_kf.append(VoVcn_kf)

                if mr.dt[i] < data_lag/2.:
                    vf_lag = lagVoVcn.calculate_tau(mr.VoVcn[0], i<1, mr.dt[i], data_lag)
                else:
                    vf_lag = mr.VoVcn[0]
                mv.VoVcn.append(vf_lag)

            plt, res, res_title = plot_P(plt, mr, mv, title + ' FP' + str(ii), Qstd=Qstd, Rstd=Rstd, data_lag=data_lag)
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



