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
    import numpy as np
    import matplotlib.pyplot as plt
    from DataOverModel import plq

    # data_file_clean = 'burstForKF_soc2p2_hi_lo_chg.csv'  # just to look at data
    data_file_clean = './noise_study/burstForKF_soc2p2_hi_lo_chg.csv'

    # Get data and statistics
    data_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)

    dt = 0.1  # Time step (seconds)
    process_noise_std = 0.1  # Standard deviation of acceleration noise
    measurement_noise_std = 0.5  # Standard deviation of voltage measurement noise
    proc_noise_std_Voa = 0.015  # volts
    proc_noise_std_Von = 0.015  # volts
    proc_noise_std_VoVcn = 0.030  # volts
    proc_noise_std_VoVca = 0.030  # volts
    proc_noise_std_Vca = 0.005  # volts
    proc_noise_std_Vcn = 0.005  # volts
    proc_noise_std_Vbv = 0.005  # volts
    proc_noise_std_Tbv = 0.005  # volts
    # Measurement noise probably 1 significant bit of converter
    meas_noise_std_Voa = 0.001  # volts
    meas_noise_std_Von = 0.001  # volts
    meas_noise_std_VoVcn = 0.002  # volts
    meas_noise_std_VoVca = 0.002  # volts
    meas_noise_std_Vca = 0.001  # volts
    meas_noise_std_Vcn = 0.001  # volts
    meas_noise_std_Vbv = 0.001  # volts
    meas_noise_std_Tbv = 0.001  # volts
    kfVoa = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                       proc_noise_std=proc_noise_std_Voa, meas_noise_std=meas_noise_std_Voa)
    kfVca = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                       proc_noise_std=proc_noise_std_Vca, meas_noise_std=meas_noise_std_Vca)
    kfVoVca = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                       proc_noise_std=proc_noise_std_VoVca, meas_noise_std=meas_noise_std_VoVca)
    kfVon = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                       proc_noise_std=proc_noise_std_Von, meas_noise_std=meas_noise_std_Von)
    kfVcn = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                       proc_noise_std=proc_noise_std_Vcn, meas_noise_std=meas_noise_std_Vcn)
    kfVoVcn = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                       proc_noise_std=proc_noise_std_VoVcn, meas_noise_std=meas_noise_std_VoVcn)
    kfVbv = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                       proc_noise_std=proc_noise_std_Vbv, meas_noise_std=meas_noise_std_Vbv)
    kfTbv = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, dt=dt,
                       proc_noise_std=proc_noise_std_Tbv, meas_noise_std=meas_noise_std_Tbv)

    run_str = 'burst data'
    ver_str = 'filtered'

    # Data structures
    mr = Saved()
    mr.time = np.array(data_raw.time)
    mr.Vca = np.array(data_raw.Vca-1.65)
    mr.Voa = np.array(data_raw.Voa-1.65)
    mr.VoVca = np.array(data_raw.VoVca)
    mr.Vcn = np.array(data_raw.Vcn-1.65)
    mr.Von = np.array(data_raw.Von-1.65)
    mr.VoVcn = np.array(data_raw.VoVcn)
    mr.Tbv = np.array(data_raw.Tbv-1.65)
    mr.Vbv = np.array(data_raw.Vbv)
    for i in range(len(data_raw.time)):
        if i == 0:
            mr.dt.append(data_raw.time[1] - data_raw.time[0])
        else:
            mr.dt.append(data_raw.time[i] - data_raw.time[i-1])

    mv = Saved()
    v_rat = None

    for i in range(len(mr.time)):
        mv.time.append(mr.time[i])

        kfVoa.predict(mr.dt[i])
        kfVoa.update(mr.Voa[i])
        vf, v_rat = kfVoa.get_state()
        mv.Voa.append(vf[0])

        kfVca.predict(mr.dt[i])
        kfVca.update(mr.Vca[i])
        vf, v_rat = kfVca.get_state()
        mv.Vca.append(vf[0])

        kfVoVca.predict(mr.dt[i])
        kfVoVca.update(mr.VoVca[i])
        vf, v_rat = kfVoVca.get_state()
        mv.VoVca.append(vf[0])

        kfVon.predict(mr.dt[i])
        kfVon.update(mr.Von[i])
        vf, v_rat = kfVon.get_state()
        mv.Von.append(vf[0])

        kfVcn.predict(mr.dt[i])
        kfVcn.update(mr.Vcn[i])
        vf, v_rat = kfVcn.get_state()
        mv.Vcn.append(vf[0])

        kfVoVcn.predict(mr.dt[i])
        kfVoVcn.update(mr.VoVcn[i])
        vf, v_rat = kfVoVcn.get_state()
        mv.VoVcn.append(vf[0])

        mv.VoVcnA.append(mv.Von[i] - mv.Vcn[i])

        kfVbv.predict(mr.dt[i])
        kfVbv.update(mr.Vbv[i])
        vf, v_rat = kfVbv.get_state()
        mv.Vbv.append(vf[0])

        kfTbv.predict(mr.dt[i])
        kfTbv.update(mr.Tbv[i])
        vf, v_rat = kfTbv.get_state()
        mv.Tbv.append(vf[0])

    plt.figure()
    plt.subplot(111)
    plt.title(' kfDemo.py var dt')
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

    plt.figure()
    plt.subplot(111)
    plt.title(' kfDemo.py var dt 2')
    plq(plt, mv, 'time', mv, 'VoVcn', color='red', linestyle='-', label='VoVcn' + ver_str)
    plq(plt, mv, 'time', mv, 'VoVcnA', color='green', linestyle='--', label='VoVcnA' + ver_str)
    plt.legend(loc=1)
    plt.figure()

    plt.subplot(111)
    plt.title(' kfDemo.py var dt 3')
    plq(plt, mr, 'time', mr, 'Voa', add=0.0, color='blue', linestyle='-', label='Voa' + run_str + '+0.0')
    plq(plt, mv, 'time', mv, 'Voa', add=0.0, color='red', linestyle='--', label='Voa' + ver_str + '+0.0')
    plq(plt, mr, 'time', mr, 'Vca', add=-0.2, color='blue', linestyle='-', label='Vca' + run_str + '-0.2')
    plq(plt, mv, 'time', mv, 'Vca', add=-0.2, color='red', linestyle='--', label='Vca' + ver_str + '-0.2')
    plq(plt, mr, 'time', mr, 'VoVca', add=-0.4, color='blue', linestyle='-', label='VoVca' + run_str + '-0.4')
    plq(plt, mv, 'time', mv, 'VoVca', add=-0.4, color='red', linestyle='--', label='VoVca' + ver_str + '-0.4')
    plt.legend(loc=1)

    plt.figure()
    plt.subplot(111)
    plt.title(' kfDemo.py var dt 3')
    plq(plt, mr, 'time', mr, 'Von', add=0.0, color='blue', linestyle='-', label='Von' + run_str + '+0.0')
    plq(plt, mv, 'time', mv, 'Von', add=0.0, color='red', linestyle='--', label='Von' + ver_str + '+0.0')
    plq(plt, mr, 'time', mr, 'Vcn', add=-0.2, color='blue', linestyle='-', label='Vcn' + run_str + '-0.2')
    plq(plt, mv, 'time', mv, 'Vcn', add=-0.2, color='red', linestyle='--', label='Vcn' + ver_str + '-0.2')
    plq(plt, mr, 'time', mr, 'VoVcn', add=-0.4, color='blue', linestyle='-', label='VoVcn' + run_str + '-0.4')
    plq(plt, mv, 'time', mv, 'VoVcn', add=-0.4, color='red', linestyle='--', label='VoVcn' + ver_str + '-0.4')
    plt.legend(loc=1)

    plt.figure()
    plt.subplot(111)
    plt.title(' kfDemo.py var dt 4')
    plq(plt, mr, 'time', mr, 'Vbv', add=-0.6, color='blue', linestyle='-', label='Vbv' + run_str + '-0.6')
    plq(plt, mv, 'time', mv, 'Vbv', add=-0.6, color='red', linestyle='--', label='Vbv' + ver_str + '-0.6')
    plq(plt, mr, 'time', mr, 'Tbv', add=-0.8, color='blue', linestyle='-', label='Tbv' + run_str + '-0.8')
    plq(plt, mv, 'time', mv, 'Tbv', add=-0.8, color='red', linestyle='--', label='Tbv' + ver_str + '-0.8')
    plt.legend(loc=1)

    plt.show(block=True)
