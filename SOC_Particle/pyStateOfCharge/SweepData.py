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
from filter.myFilters import LagExp
from scipy.signal import find_peaks
from itertools import pairwise
from DataOverModel import write_clean_file
from plot.plq import plq as plq

class SavedDataSweep:
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
        Iterates over members of a self, assigns values to numpy.ndarray members
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
def load_data(path_to_data, time_end):

    print(f"load_data_KF1x1_test: \n{path_to_data=}\n{time_end=}\n")

    hdr_key_x = "unit_shunt,"  # Find one self of title
    unit_key_x = "shunt_unit"

    data_file_clean = write_clean_file(path_to_data, type_='_shunt', hdr_key=hdr_key_x, unit_key=unit_key_x)
    if data_file_clean is None:
        return None, None
    import numpy as np
    if data_file_clean is not None:
        mon_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        mon_raw = None
        print(f"load_data_KF1x1_test: returning mon=None")

    mon = SavedDataSweep(x=mon_raw, time_end=time_end)

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
            # print(f"{steady_level_lag=} {std_dev_lag=}")
            print(f"{index_start_sweep_lag=} {time_start_sweep_lag=}")
            print(f"{index_end_sweep_lag=} {time_end_sweep_lag=}")
        except IndexError:
            steady_only = True


# Example Usage:  Ran ok on 20260217
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from plot.plq import plq as plq
    plt.rcParams['axes.grid'] = True
    plt.rcParams['legend.fontsize'] = 'small'
    from filter.butterHighPassDemo import butter_highpass_filter
    time_end = None
    from filter.KF1x1_test import load_data_KF1x1_test

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
    data_file = 'C:/Users/daveg/Documents/GitHub/mySolarStateOfCharge/SOC_Particle/pyStateOfCharge/noise_study/sweepchirp4_soc2p2_hi_lo_chg.csv'  # Cx46000, new base 20251231
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
