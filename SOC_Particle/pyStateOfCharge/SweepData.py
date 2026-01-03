# SweepData:  Filter swept sine data
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

import numpy as np
from myFilters import LagExp
from scipy.signal import find_peaks
from itertools import pairwise
from DataOverModel import write_clean_file, plq

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
            self.skip_x = np.bool(np.array(x.skip))
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
def load_data(path_to_data, time_end_in):

    print(f"load_data_KF1x1_test: \n{path_to_data=}\n{time_end_in=}\n")

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

    mon = SavedData(x=mon_raw, time_end=time_end_in)

    return mon, data_file_clean


class Saved:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self):
        self.time = []
        self.dt = []
        self.VoVcn = []
        self.VoVcn_kf = []
        self.VoVcn_filt = []




"""Manage swept sine data."""


class Wave:
    """Periodic (sinusoidal) wave data class."""

    def __init__(self, datav=np.array([0., 0.]), timev=np.array([0., 1.]), sample_hz=1.,
                 ss_time_rng=np.array([0., 1.]), tool_lag=None,
                 data_lag_nyquist_ratio=50., sigma_factor=6., sweep_s=None):
        """
        Initializes a signal, probably sinusoidal
        Args:
            datav (float): Vector of signal
            timev (float):  Vector of time corresponding to data points
            sample_hz (float):  Assumed constant sample frequency, Hz
            ss_time_rng ([ tss_start, tss_end ])
                tss_start (float):  Specified low end of reliable steady vector metrics, s
                tss_end (float):  Specified high end of reliable steady vector metrics, s
            tool_lag (float):
            data_lag_nyquist_ratio (float):  multiples of Nyquist to avoid aliasing and still achieve a minimum fidelity
        """
        self.t = timev
        self.x = datav
        self.ss_time_rng = ss_time_rng


        # Some initial screening
        self.vec_initial = np.where((self.t <= ss_time_rng[1]) & (self.t >= ss_time_rng[0]))
        self.x_avg = np.average(self.x[self.vec_initial])
        self.x = self.x - self.x_avg
        self.x_max = max(self.x)
        self.x_min = min(self.x)
        self.N = len(self.t)
        self.total_time = self.t[-1] - self.t[0]
        self.sample_hz = float(self.N) / self.total_time
        self.sample_s = 1. / self.sample_hz
        self.sample_rps = self.sample_hz * 2. * np.pi
        self.nyquist_rps = self.sample_rps / 2.
        if tool_lag is None:
            min_possible_data_lag = 0.07 / self.nyquist_rps * data_lag_nyquist_ratio
            self.data_lag = min_possible_data_lag * 5.
        self.ToolLag = LagExp(dt=self.sample_s, tau=self.data_lag, max_=self.x_max, min_=self.x_min)
        self.x_lag = None
        self.std_dev_x_lag = None
        self.sigma_factor = sigma_factor
        self.sweep_s = sweep_s
        self.dt = [b - a for a, b in pairwise(self.t)]
        self.dt.insert(0, self.dt[0])
        self.x_lag = [self.ToolLag.calculate(self.x[i], reset=i < 1, dt=self.dt[i]) for i in range(N)]


        self.x_lag_rate = np.gradient(self.x_lag, self.t)
        imax = [i+1 for i, _ in enumerate(find_peaks(self.x_lag_rate)[0])]

        # Detect positive zero crossings
        self.x_lag = np.array(self.x_lag)
        std_dev_lag = np.std(self.x_lag[self.vec_initial])
        index_start_sweep_lag = np.array(np.where(self.x_lag < -10.*std_dev_lag))[0, 0]
        time_start_sweep_lag = self.t[index_start_sweep_lag]
        index_end_sweep_lag = np.where(self.t < time_start_sweep_lag + 650.)[0][-1]
        time_end_sweep_lag = self.t[index_end_sweep_lag]
        is_positive = self.x_lag[index_start_sweep_lag:index_end_sweep_lag]  > 0
        positive_crossings = (~is_positive[:-1]) & is_positive[1:]
        self.crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep_lag  # Add 1 to account for the shift
        self.time_zero_crossing = self.t[self.crossing_indices]

        self.dtime_zero_crossing = [b - a for a, b in pairwise(self.time_zero_crossing)]
        self.dtime_zero_crossing.insert(0, self.dtime_zero_crossing[0])
        self.sampling_hz = [1./self.dtime_zero_crossing[i] for i in range(len(self.dtime_zero_crossing))]

        # Detect the excitation frequency as a function of time

        plt.figure()
        print("plot_1:", end='')
        plt.subplot(211)
        plt.title('SweepDataWave' + '1')
        plq(plt, self, 'time_zero_crossing', self, 'dtime_zero_crossing', color='blue', linestyle='-', label='dtime_zero_crossing')
        plt.text(0.5, 0.2, f"tool_lag={self.data_lag} ",
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 fontsize=12,
                 color='blue',
                 bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
        plt.legend(loc=1)
        plt.subplot(212)
        plq(plt, self, 'time_zero_crossing', self, 'sampling_hz', color='red', linestyle='-', label='sampling_hz')
        plt.legend(loc=1)

        plt.figure()
        print("plot_2:", end='')
        plt.subplot(211)
        plt.title('SweepDataWave' + '2')
        plq(plt, self, 't', self, 'x', color='blue', linestyle='-', label='x')
        plq(plt, self, 't', self, 'x_lag', color='red', linestyle='--', label='x_lag')
        plt.text(0.5, 0.2, f"tool_lag={self.data_lag} ",
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 fontsize=12,
                 color='blue',
                 bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
        plt.legend(loc=1)
        plt.subplot(212)
        plq(plt, self, 't', self, 'x_lag_rate', color='red', linestyle='-', label='x_lag_rate')
        plt.legend(loc=1)

        plt.show()

    def lag(self):
        self.x_lag = []
        for i in range(self.N):
            lagged_val = self.ToolLag.calculate(self.x[i], reset=i < 1, dt=0.1)  # Actual dt req'd at run. 0.1 is placeholder
            self.x_lag.append(lagged_val)
        self.x_lag = np.array(self.x_lag)- self.x_avg
        self.std_dev_x_lag = np.std(self.x_lag[self.vec_initial])  # For finding start of sweep

        try:
            index_start_sweep_lag = np.array(np.where( self.x_lag < -self.sigma_factor*self.std_dev_x_lag))[0, 0]
            time_start_sweep_lag = self.t[index_start_sweep_lag]
            index_end_sweep_lag = np.where(self.t < time_start_sweep_lag + self.sweep_s)[0][-1]
            time_end_sweep_lag = mr.time[index_end_sweep_lag]
            print(f"{steady_level_lag=} {std_dev_lag=}")
            print(f"{index_start_sweep_lag=} {time_start_sweep_lag=}")
            print(f"{index_end_sweep_lag=} {time_end_sweep_lag=}")
        except IndexError:
            steady_only = True

    # # Recenter self.x for freq analysis:  assume at least 179 sec fr
    # # vec_fr = np.arange(index_start_sweep_lag, index_end_sweep_lag)
    # if not steady_only:
    #     vec_fr_for_avg = np.arange(index_start_sweep_lag + int(0.5*(index_end_sweep_lag - index_start_sweep_lag)),
    #                                index_end_sweep_lag)
    # else:
    #     vec_fr_for_avg = vec_initial
    #
    # # Center
    # self.x_avg = np.average(self.x[vec_fr_for_avg])
    # self.x = self.x - self.x_avg
    # mr.VoVcn_kf = np.array(mr.VoVcn_kf - self.x_avg)
    # mr.VoVcn_filt = np.array(mr.VoVcn_filt - self.x_avg)
    # self.x_lag = np.array(self.x_lag - self.x_avg)
    #
    # mv.VoVcn_kf = np.array(mv.VoVcn_kf)
    # mv.VoVcn_kf_avg = np.average(mv.VoVcn_kf[vec_fr_for_avg])
    # mv.VoVcn_kf = mv.VoVcn_kf - mv.VoVcn_kf_avg
    # mv.VoVcn_kf_avg = np.average(mv.VoVcn_kf[vec_fr_for_avg])  # recalculate
    #
    # mv.VoVcn_kf_avg = np.full((len(mv.VoVcn_kf),), mv.VoVcn_kf_avg)
    #
    #
    # # steady_only = True
    # if not steady_only:
    #     # Detect positive zero crossings
    #     is_positive = self.x_lag[index_start_sweep_lag:index_end_sweep_lag]  > 0
    #     positive_crossings = (~is_positive[:-1]) & is_positive[1:]
    #     mr.crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep_lag  # Add 1 to account for the shift
    #     mr.time_zero_crossing = mr.time[mr.crossing_indices]
    #
    #     mv.time = mr.time
    #     is_positive = mv.VoVcn_kf[index_start_sweep_lag:index_end_sweep_lag]  > 0
    #     positive_crossings = (~is_positive[:-1]) & is_positive[1:]
    #     mv.crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep_lag  # Add 1 to account for the shift
    #     mv.time_zero_crossing = mv.time[mv.crossing_indices]
    #
    #     # For simplicity assume mv zero crossing is always after mr zero crossing (lags behave like lags)
    #     # This also implies minimum phase behavior (magnitude decreasing) so normalize to max
    #     print("Time:    Frequency, Hz  /  Magnitude, dB   /  Phase, deg / raw_lag(s) / data_lag_lag(deg) / data_lag_lag(s) / lag(s)")
    #     transfer_function = []
    #     mag_normal = 0.
    #     for j in range(len(mr.time_zero_crossing)-1):
    #         time = mr.time_zero_crossing[j]
    #         index = mr.crossing_indices[j]
    #         period = mr.time_zero_crossing[j+1] - mr.time_zero_crossing[j]
    #         frequency = 1. / period
    #         ang_freq = frequency * 2. * np.pi
    #         data_lag_lag_deg = np.atan2(data_lag*ang_freq, 1.) * 180./np.pi
    #         data_lag_lag = data_lag_lag_deg / 360. * period
    #         raw_lag = mv.time_zero_crossing[j] - mr.time_zero_crossing[j]
    #         lag = raw_lag + data_lag_lag
    #         input = self.x_lag[mr.crossing_indices[j]:mr.crossing_indices[j+1]]
    #         input_magnitude = max(input) - min(input)
    #         if j >= len(mv.crossing_indices) - 1:
    #             break
    #         response = mv.VoVcn_kf[mv.crossing_indices[j]:mv.crossing_indices[j+1]]
    #         response_magnitude = (max(response)[0] - min(response)[0])
    #         tf_magnitude = 20.*np.log10(response_magnitude/input_magnitude)
    #         tf_phase = -360. * lag / period
    #         if frequency < 2.5:
    #             mag_normal = max(mag_normal, tf_magnitude)
    #         transfer_function.append([frequency, tf_magnitude, tf_phase, time, raw_lag, data_lag_lag_deg,
    #                                   data_lag_lag, lag, index])
    #         # print(f"{frequency}  /  {tf_magnitude}    / {tf_phase}")
    #     transfer_function = np.array(transfer_function)
    #     transfer_function[:, 1] -= mag_normal
    #
    #     # Cleanup the result
    #     d = np.diff(transfer_function, axis=0)[:, 0]
    #     pos_indeces = np.where(d > 0.)
    #     transfer_function = transfer_function[pos_indeces]
    #     # Go through one-by-one and delete bad steps
    #     tf_clean = []
    #     tf_clean.append(transfer_function[0, :])
    #     k_clean = 0
    #     n = len(transfer_function[:, 0])
    #     k = 1
    #     while k < n:
    #         if (abs(transfer_function[k, 0] - tf_clean[k_clean][0]) < 1. and
    #                 (transfer_function[k, 0] - tf_clean[k_clean][0]) > 0.):
    #             tf_clean.append(transfer_function[k, :])
    #             k_clean += 1
    #         k += 1
    #     tf_clean = np.array(tf_clean)
    #     for j in range(len(tf_clean)):
    #         print("{:8.2f}: ".format(tf_clean[j][3]),
    #               "{:7.1f} Hz / ".format(tf_clean[j][0]),
    #               "{:7.1f} dB / ".format(tf_clean[j][1]),
    #               "{:7.1f} deg / ".format(tf_clean[j][2]),
    #               "{:7.3f} s  / ".format(tf_clean[j][4]),
    #               "{:7.1f} deg  / ".format(tf_clean[j][5]),
    #               "{:7.3f} s  / ".format(tf_clean[j][6]),
    #               "{:7.3f} s  / ".format(tf_clean[j][7]),
    #               )
    #     mv.time_clean = tf_clean[:, 3]
    #     mv.f_clean = tf_clean[:, 0]
    #     mv.w_clean = mv.f_clean * 2. * np.pi
    #     mv.mdB_clean = tf_clean[:, 1]
    #     mv.phs_clean = tf_clean[:, 2]
    #
    # # Metrics
    # mr.VoVcn_steady = self.x[vec_initial]
    # mv.VoVcn_kf_steady = mv.VoVcn_kf[vec_initial]
    # mr.VoVcn_steady_lag = self.x_lag[vec_initial]
    # mr.amp_VoVcn_steady = np.max(mr.VoVcn_steady) - np.min(mr.VoVcn_steady)
    # mv.amp_VoVcn_kf_steady = np.max(mv.VoVcn_kf_steady) - np.min(mv.VoVcn_kf_steady)
    # mr.amp_VoVcn_steady_lag = np.max(mr.VoVcn_steady_lag) - np.min(mr.VoVcn_steady_lag)
    # print(f" amp VoVcn_kf_steady  {mv.amp_VoVcn_kf_steady}   amp VoVcn_steady {mr.amp_VoVcn_steady}" )
    # attenuation = mv.amp_VoVcn_kf_steady / mr.amp_VoVcn_steady
    # attenuation_lag = mr.amp_VoVcn_steady_lag / mr.amp_VoVcn_steady
    #
    # if not steady_only:
    #     phase45_tf_index = 0
    #     phase90_tf_index = 0
    #     db3_tf_index = 0
    #     for j in range(len(tf_clean)-1):
    #         frequency = tf_clean[j][0]
    #         phase_dg = tf_clean[j][2]
    #         mag_db = tf_clean[j][1]
    #         time = tf_clean[j][3]
    #         if mag_db < tf_clean[db3_tf_index][1] and mag_db >= -3.:
    #             db3_tf_index = j
    #         if  phase_dg < tf_clean[phase45_tf_index][2] and phase_dg >= -45.:
    #             phase45_tf_index =j
    #         if phase_dg < tf_clean[phase90_tf_index][2] and phase_dg >= -90.:
    #             phase90_tf_index = j
    #     freq_3db = tf_clean[db3_tf_index][0]
    #     tau_3db = 1. / (freq_3db * 2. * np.pi)
    #     mag_3db = tf_clean[db3_tf_index][1]
    #     time_3db = tf_clean[db3_tf_index][3]
    #     freq_45 = tf_clean[phase45_tf_index][0]
    #     tau_45 = 1. / (freq_45 * 2. * np.pi)
    #     phase_45 = tf_clean[phase45_tf_index][2]
    #     time_45 = tf_clean[phase45_tf_index][3]
    #
    #     freq_90 = tf_clean[phase90_tf_index][0]
    #     omega_90 = freq_90 * 2. * np.pi
    #     tau_90 = 1. / (freq_90 * 2. * np.pi)
    #     phase_90 = tf_clean[phase90_tf_index][2]
    #     time_90 = tf_clean[phase90_tf_index][3]
    #
    #
    #     print(f"{attenuation=} {attenuation_lag=}")
    #     print(f"{time_3db=} {freq_3db=} {mag_3db=}")
    #     print(f"{time_45=}  {freq_45=} {phase_45=}")
    #     print(f"{time_90=}  {freq_90=} {phase_90=} {omega_90=}")
    # metric_string = "Metrics:\n"
    # metric_string += "  Qstd = {:9.6f}\n  Rstd =     {:9.6f}\n  data_lag = {:7.4f}\n\n".format(Qstd, Rstd, data_lag)
    # metric_string += "  Attn = {:5.2f}  Attn_lag = {:5.2f}\n\n".format(attenuation, attenuation_lag)
    # if not steady_only:
    #     metric_string += "  -3db @    {:4.2f} Hz,  ({:5.1f} sec)\n".format(freq_3db, time_3db)
    #     metric_string += "  -45 deg @ {:4.2f} Hz   ({:5.1f} sec)\n\n".format(freq_45, time_45)
    #     metric_string += "  -90 deg @ {:4.2f} Hz   ({:5.1f} sec)\n\n".format(freq_90, time_90)
    #     metric_string += "  tau @ -3db = {:5.3f}\n  tau @ -45 = {:5.3f}\n  omega90 = {:5.3f}\n".format(tau_3db, tau_45, omega_90)
    #     res_title = "Qstd, Rstd, data_lag, attenuation_lag, amp_steady_kf, amp_steady, attenuation, tau_3db, tau_45, omega_90,"
    #     res = [Qstd, Rstd, data_lag, attenuation_lag, mv.amp_VoVcn_kf_steady, mr.amp_VoVcn_steady, attenuation, tau_3db, tau_45, omega_90]
    # else:
    #     res_title = "Qstd, Rstd, data_lag, attenuation_lag,  amp_steady_kf, amp_steady, attenuation, tau_3db, tau_45, omega_90,"
    #     res = [Qstd, Rstd, data_lag, attenuation_lag, mv.amp_VoVcn_kf_steady, mr.amp_VoVcn_steady, attenuation,  0., 0., 0.]


# Example Usage:
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from DataOverModel import plq
    plt.rcParams['axes.grid'] = True
    from butterHighPassDemo import butter_highpass_filter
    time_end = None
    from KF1x1_test import load_data_KF1x1_test

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
    data_file = './noise_study/sweepchirp4_soc2p2_hi_lo_chg.csv'  # Cx46000, new base 20251231
    doing_doe = True  # Toggle this to see various kf implemented in python
    cutoff_freq_hz = 0.05  # hpf
    # The best design of filter
    Qstd = 0.0003  # Standard deviation of acceleration noise
    Rstd = 0.0100  # Standard deviation of voltage measurement noise



############################################################33
    mr, data_file_clean = load_data(data_file, time_end)
    title = 'VoVc Base KF1x1_test.py var dt'
    N = len(mr.time)
    total_time = mr.time[-1]
    sample_hz = float(N) / total_time

    dt = 0.1  # Time step (seconds) used only on init
    WaveVoVcn = Wave(mr.vovcn, mr.time, sample_hz=sample_hz, ss_time_rng=[10., 50.])
