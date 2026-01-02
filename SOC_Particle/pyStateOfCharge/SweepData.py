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

"""Manage swept sine data."""


class Wave:
    """Periodic (sinusoidal) wave data class."""

    def __init__(self, data=[0., 0.], timev=[0., 1.], sample_hz=1., ss_time_rng=[0., 1.], tool_lag=None,
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
        self.x = data
        self.ss_time_rng = ss_time_rng

        # Some initial screening
        self.vec_initial = np.where( (ss_time_rng[0] <= self.t <= ss_time_rng[1]) )
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
        self.lag()

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

    # Recenter self.x for freq analysis:  assume at least 179 sec fr
    # vec_fr = np.arange(index_start_sweep_lag, index_end_sweep_lag)
    if not steady_only:
        vec_fr_for_avg = np.arange(index_start_sweep_lag + int(0.5*(index_end_sweep_lag - index_start_sweep_lag)),
                                   index_end_sweep_lag)
    else:
        vec_fr_for_avg = vec_initial

    # Center
    self.x_avg = np.average(self.x[vec_fr_for_avg])
    self.x = self.x - self.x_avg
    mr.VoVcn_kf = np.array(mr.VoVcn_kf - self.x_avg)
    mr.VoVcn_filt = np.array(mr.VoVcn_filt - self.x_avg)
    self.x_lag = np.array(self.x_lag - self.x_avg)

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
        is_positive = self.x_lag[index_start_sweep_lag:index_end_sweep_lag]  > 0
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
            input = self.x_lag[mr.crossing_indices[j]:mr.crossing_indices[j+1]]
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
    mr.VoVcn_steady = self.x[vec_initial]
    mv.VoVcn_kf_steady = mv.VoVcn_kf[vec_initial]
    mr.VoVcn_steady_lag = self.x_lag[vec_initial]
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

