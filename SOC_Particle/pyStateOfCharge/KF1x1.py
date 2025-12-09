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
from myFilters import LagTustin

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
    plq(plt, mv, 'time', mv, 'Von', add=0.0, color='red', linestyle='--', label='Von' + ver_str + '+0.0')
    plq(plt, mr, 'time', mr, 'Vcn', add=-0.2, color='blue', linestyle='-', label='Vcn' + run_str + '-0.2')
    plq(plt, mv, 'time', mv, 'Vcn', add=-0.2, color='red', linestyle='--', label='Vcn' + ver_str + '-0.2')
    plq(plt, mr, 'time', mr, 'VoVcn', add=-0.4, color='blue', linestyle='-', label='VoVcn' + run_str + '-0.4')
    plq(plt, mv, 'time', mv, 'VoVcn', add=-0.4, color='red', linestyle='--', label='VoVcn' + ver_str + '-0.4')
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
    plq(plt, mv, 'time', mv, 'Von', add=0.0, color='red', linestyle='--', label='Von' + ver_str + '+0.0')
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

def plot_P(plt=None, mr=None, mv=None, title=None, Qstd=None, R=None):
    N = len(mr.Von)
    total_time = mr.time[-1]
    sample_freq_hz = float(N) / total_time
    sample_time = 1. / sample_freq_hz
    sample_freq_rps = sample_freq_hz * 2. * np.pi
    nyquist_freq_rps = sample_freq_rps / 2.
    lpf_tau = 0.07 / nyquist_freq_rps * 50.
    mr_lpf = LagTustin(dt=sample_time, tau=lpf_tau, max_=5., min_=-5.)
    mv_lpf = LagTustin(dt=sample_time, tau=lpf_tau, max_=5., min_=-5.)

    print(f"{sample_freq_hz=} {lpf_tau=}")

    # Filter signal for cleaner statistical testing
    mr.Von_lpf = []
    mv.Von_lpf = []
    mv.dt = mr.dt
    for i in range(N):
        if i == 0:
            reset = True
        else:
            reset = False
        mr.Von_lpf.append(mr_lpf.calculate(mr.Von[i], reset=reset, dt=mr.dt[i]))
        mv.Von_lpf.append(mv_lpf.calculate(mv.Von[i], reset=reset, dt=mv.dt[i]))
    mr.Von_lpf = np.array(mr.Von_lpf)
    mv.Von_lpf = np.array(mv.Von_lpf)

    # Get initial steady offset so can search for start of sweep.  Assume initial 50 seconds are steady.
    vec_initial = np.where(mr.time <= 50.)
    steady_level = np.average(mr.Von[vec_initial])
    mr.Von = mr.Von - steady_level
    std_dev = np.std(mr.Von[vec_initial])
    index_start_sweep = np.array(np.where( abs(mr.Von) > 5.*std_dev ))[0][0]
    time_start_sweep = mr.time[index_start_sweep]
    print(f"{steady_level=} {std_dev=} {index_start_sweep=} {time_start_sweep=}")
    std_dev_lpf = np.std(mr.Von_lpf[vec_initial])
    steady_level_lpf = np.average(mr.Von_lpf[vec_initial])
    mr.Von_lpf = mr.Von_lpf - steady_level
    index_start_sweep_lpf = np.array(np.where( abs(mr.Von_lpf) > 3.*std_dev_lpf))[0][0]
    time_start_sweep_lpf = mr.time[index_start_sweep_lpf]
    print(f"{steady_level_lpf=} {std_dev_lpf=} {index_start_sweep_lpf=} {time_start_sweep_lpf=}")

    # Detect positive zero crossings
    is_positive = mr.Von_lpf[index_start_sweep_lpf:-1]  > 0
    positive_crossings = (~is_positive[:-1]) & is_positive[1:]
    crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep  # Add 1 to account for the shift
    time_zero_crossing = mr.time[crossing_indices]
    print(f"mr  {time_zero_crossing=}")

    is_positive = mv.Von_lpf[index_start_sweep_lpf:-1]  > 0
    positive_crossings = (~is_positive[:-1]) & is_positive[1:]
    crossing_indices = np.where(positive_crossings)[0] + 1 + index_start_sweep  # Add 1 to account for the shift
    time_zero_crossing = mr.time[crossing_indices]
    print(f"mv  {time_zero_crossing=}")


    index_end_sweep_lpf = np.array(np.where( abs(mr.Von_lpf[index_start_sweep_lpf:-1] - steady_level_lpf) <= 3.*std_dev_lpf))[0][0]


    window = 20  # 5 Hz wiggle @ 40 Hz sampling X 2
    mr.Von_rms = running_rms(np.array(mr.Von), window_size=window)
    mv.Von_rms = running_rms(np.array(mv.Von), window_size=window)

    mv.Von_atten = 20 * np.log10(mv.Von_rms / mr.Von_rms)
    # mv.Voa_atten = 20 * np.log10(mv.Voa_rms / mr.Voa_rms)

    plt.figure()
    plt.subplot(311)
    plt.title(title)
    plq(plt, mr, 'time', mr, 'Von', color='blue', linestyle='-', label='Von' + run_str)
    plq(plt, mv, 'time', mv, 'Von', color='red', linestyle='--', label='Von' + ver_str)
    plt.text(0.5, 0.5, f"{Qstd=}   {R=}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=14,
             color='blue',
             bbox=dict(facecolor='yellow', alpha=0.5, pad=5))
    plt.legend(loc=1)
    plt.subplot(312)
    plq(plt, mr, 'time', mr, 'Von_lpf', color='blue', linestyle='-', label='Von_lpf' + run_str)
    plq(plt, mv, 'time', mv, 'Von_lpf', color='red', linestyle='--', label='Von_lpf' + ver_str)
    plt.legend(loc=1)
    plt.subplot(313)
    plq(plt, mr, 'time', mr, 'Von_rms', color='blue', linestyle='-', label='Von_rms' + run_str)
    plq(plt, mv, 'time', mv, 'Von_rms', color='red', linestyle='--', label='Von_rms' + ver_str)
    plt.legend(loc=1)

    return plt

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
        self.Bu = 0.  # Control transition
        # Process noise covariance matrix (assuming noise in acceleration)
        self.Q_std = proc_noise_std
        self.R = np.array([[meas_noise_std**2]])  # State uncertainty.  Measurement noise covariance matrix
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
        s += "  R = {:10.6g}\n".format(self.R[0][0])
        s += "  Q_std = {:10.6g}\n".format(self.Q_std)
        s += "  H = " + self.H.__str__() + "\n"
        s += "  Outputs:\n"
        s += "  x  = \n" + self.x.__str__() + "\n"
        s += "  hx = {:10.6g}\n".format(self.hx)
        s += "  y  = {:10.6g}\n".format(self.y_kf[0][0])
        s += "  P  = \n" + self.P.__str__() + "\n"
        s += "  K  = \n" + self.K.__str__() + "\n"
        s += "  S  = {:10.6g}\n".format(self.S[0][0])
        return s

    def predict(self, dt):
        """
        Performs the prediction step of the Kalman filter.
        Inputs:
            u   1x1 input, =ib, A
            Bu  1x1 control transition, Ohms
            Fx  1x1 state transition, V/V
        Outputs:
            x   1x1 Kalman state variable = Vsoc (0-1 fraction)
            P   1x1 Kalman probability
        """
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.Fx = np.array([[1.0, self.dt], [0.0, 1.0]])

        # Process noise covariance matrix (assuming noise affects acceleration)
        G = np.array([[0.5 * self.dt ** 2], [dt]])
        Q = G @ G.T * self.Q_std ** 2

        # Predict state and covariance
        self.x = self.Fx @ self.x
        self.P = self.Fx @ self.P @ self.Fx.T + Q

    def update(self, measurement):
        """
        Performs the update step of the Kalman filter.
        Updates the state estimate and covariance matrix based on the new measurement.

        Args:
            measurement (float): The new position measurement.
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

    def update_kf(self, measurement):
        """
        Performs the update step of the Kalman filter.
        Updates the state estimate and covariance matrix based on the new measurement.

        Args:
            measurement (float): The new position measurement.
        """
        # Kalman Gain
        self.S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(self.S)

        # Update state estimate
        self.y_kf = measurement - (self.H @ self.x)  # Innovation
        self.x = self.x + (self.K @ self.y_kf)

        # Update covariance matrix
        self.P = (np.eye(self.x.shape[0]) - self.K @ self.H) @ self.P

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
            self.Vca = None
            self.Voa = None
            self.VoVca = None
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


# Example Usage:
if __name__ == "__main__":
    """
    Test setup:  FY6900 Dominty Function Generator.  CH 1 connected across shunt leads.  CH 2 ground connected to
    board ground.  Top level - Sweep.   - Freq 0.5 - 5.0, Ampl 0.01 - 0.01, Offs -0.01 - -0.01, Duty 50% - 50%,
    Mode Logarithm.   Direction Forth, Time 120s.  Turn off generator.
    'Cx27000',  wait 60 sec. Turn on generator and press OK on function generator.  When it reaches 0.5 Hz again press
    OK to stop.  Then turn off generator.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from DataOverModel import plq
    plt.rcParams['axes.grid'] = True

    time_end = None
    data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\burstForKF_soc2p2_hi_lo_chg.csv'
    mr, data_file_clean = load_data(data_file, time_end)
    title = 'Base kfDemo.py var dt'
    dt = 0.1  # Time step (seconds) used only on init
    Qstd = 0.015*2  # Standard deviation of acceleration noise
    R = 0.001  # Standard deviation of voltage measurement noise

    have_Vca = False
    have_Voa = False
    have_Vcn = False
    have_Von = False
    have_VoVca = False
    have_VoVca = False
    have_Tbv = False
    have_Vbv = False

    if hasattr(mr, 'Vca'):
        have_Vca = True
        mr.Vca -= 1.65
        kfVca = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'Voa'):
        have_Voa = True
        mr.Voa -= 1.65
        kfVoa = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'Vcn'):
        have_Vcn = True
        mr.Vcn -= 1.65
        kfVcn = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'Von'):
        have_Von = True
        mr.Von -= 1.65
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
        mr.Tbv -= 1.65
        kfTbv = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)
    if hasattr(mr, 'Vbv'):
        have_Vbv = True
        kfVbv = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)

    run_str = '_burst data'
    ver_str = '_filtered'

    title = 'Part Factorial kfDemo.py var dt'
    # for Qstd, R in [[0.015, 0.001], [0.03, 0.001], [0.0075, 0.001], [0.015, 0.002], [0.015, 0.0005],
    #                 [0.015, 0.0005], [0.03, 0.0005], [0.0075, 0.0005], [0.015, 0.001], [0.015, 0.00025]]:
    for Qstd, R in [[0.015, 0.001],  [0.015, 0.00025]]:
        mv = None
        print(f"{Qstd=} {R=}")
        kfVon = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                           proc_noise_std=Qstd, meas_noise_std=R)

        run_str = '_burst data'
        ver_str = '_filtered'

        # Data structures
        mv = Saved()
        v_rat = None

        for i in range(len(mr.time)):
            mv.time.append(mr.time[i])

            kfVon.predict(mr.dt[i])
            kfVon.update(mr.Von[i])
            vf, v_rat = kfVon.get_state()
            mv.Von.append(vf[0])

        plt = plot_P(plt, mr, mv, title + ' P1', Qstd=Qstd, R=R)


    # General purpose results
    # Data structures
    if have_Vca and have_VoVca and have_Tbv and have_Vbv and have_VoVcn and have_Vcn and have_Von:
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

    plt.show(block=True)



