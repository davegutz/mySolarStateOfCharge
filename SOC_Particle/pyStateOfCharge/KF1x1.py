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

import numpy as np

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

    def __init__(self, initial_position, initial_velocity, proc_noise_std, meas_noise_std):
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
        s += "  Fx = {:13.10g}\n".format(self.Fx)
        s += "  Bu = {:13.10g}\n".format(self.Bu)
        s += "  R = {:10.6g}\n".format(self.R)
        s += "  Q_std = {:10.6g}\n".format(self.Q_std)
        s += "  H = {:10.6g}\n".format(self.H)
        s += "  Outputs:\n"
        s += "  x  = {:10.6g}\n".format(self.x)
        s += "  hx = {:10.6g}\n".format(self.hx)
        s += "  y  = {:10.6g}\n".format(self.y_kf)
        s += "  P  = {:10.6g}\n".format(self.P)
        s += "  K  = {:10.6g}\n".format(self.K)
        s += "  S  = {:10.6g}\n".format(self.S)
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
        # State transition matrix (constant velocity model)
        self.Fx = np.array([[1.0, dt], [0.0, 1.0]])

        # Process noise covariance matrix (assuming noise affects acceleration)
        G = np.array([[0.5 * dt ** 2], [dt]])
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


# Example Usage:
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from DataOverModel import plq

    N = 100
    dt = 0.1  # Time step (seconds)
    process_noise_std = 0.1  # Standard deviation of acceleration noise
    measurement_noise_std = 0.5  # Standard deviation of position measurement noise
    kf3 = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, proc_noise_std=process_noise_std,
                     meas_noise_std=measurement_noise_std)
    kf4 = KF1x1VarDt(initial_position=0.0, initial_velocity=0.0, proc_noise_std=process_noise_std,
                     meas_noise_std=measurement_noise_std)
    mr3 = Saved()
    mv3 = Saved()
    mr4 = Saved()
    mv4 = Saved()

    # Simulate some measurements
    true_position3 = 0.0
    true_velocity3 = 1.0  # Constant velocity
    true_position4 = 0.0
    true_velocity4 = 0.0  # Constant velocity
    t = 0
    for i in range(N):
        dt_noise = max(np.random.normal(dt, 0.005), .0001)
        t += dt + dt_noise
        true_position3 += true_velocity3 * dt_noise
        # Add some random noise to the measurement
        noise = np.random.normal(0, measurement_noise_std)
        noisy_measurement3 = true_position3 + noise
        noisy_measurement4 = true_position4 + noise
        mr3.dt.append(dt_noise)
        mr3.pos.append(noisy_measurement3)
        mr3.time.append(t)
        mr3.dt.append(dt_noise)
        mr3.velo.append(true_velocity3)
        mr4.dt.append(dt_noise)
        mr4.pos.append(noisy_measurement4)
        mr4.time.append(t)
        mr4.dt.append(dt_noise)
        mr4.velo.append(true_velocity4)

    n = len(mr3.pos)
    for i in range(n):
        x = mr3.pos[i]
        dt = mr3.dt[i]
        kf3.predict(dt)
        kf3.update(x)
        pos, vel = kf3.get_state()
        mv3.pos.append(pos)
        mv3.velo.append(vel)
    mv3.time = mr3.time
    mv3.dt = mr3.dt
    n = len(mr4.pos)
    for i in range(n):
        x = mr4.pos[i]
        dt = mr4.dt[i]
        kf4.predict(dt)
        kf4.update(x)
        pos, vel = kf4.get_state()
        mv4.pos.append(pos)
        mv4.velo.append(vel)
    mv4.time = mr4.time
    mv4.dt = mr4.dt

    run_str3 = 'data 1 var dt'
    ver_str3 = 'filtered 1 var dt'
    run_str4 = 'data 2 var dt'
    ver_str4 = 'filtered 2 var dt'

    plt.figure()
    plt.subplot(121)
    plt.title(' kfDemo.py var dt')
    plq(plt, mr3, 'time', mr3, 'pos', color='blue', linestyle='-', label='pos1' + run_str3)
    plq(plt, mv3, 'time', mv3, 'pos', color='red', linestyle='--', label='pos1' + ver_str3)
    plq(plt, mr4, 'time', mr4, 'pos', color='magenta', linestyle='-', label='pos2' + run_str4)
    plq(plt, mv4, 'time', mv4, 'pos', color='black', linestyle='--', label='pos2' + ver_str4)
    plt.legend(loc=1)
    plt.subplot(122)
    plq(plt, mr3, 'time', mr3, 'velo', color='blue', linestyle='-', label='velo' + run_str3)
    plq(plt, mv3, 'time', mv3, 'velo', color='red', linestyle='--', label='velo' + ver_str3)
    plq(plt, mr4, 'time', mr4, 'velo', color='magenta', linestyle='-', label='velo' + run_str4)
    plq(plt, mv4, 'time', mv4, 'velo', color='black', linestyle='--', label='velo' + ver_str4)
    plt.legend(loc=1)

    plt.show(block=True)
