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
import numpy as np

"""1x1 General Purpose Kalman Filter.   Inherit from this class and include kf_predict and
kf_update methods in the parent."""


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
        self.COLS = 2
        self.ROWS = 2
        self.dt = dt
        self.Fx = np.array([[1.0, self.dt], [0.0, 1.0]])  # State transition
        self.G  = None
        self.Bu = 0.  # Control transition
        # Process noise covariance matrix (assuming noise in acceleration)
        self.Q_std = proc_noise_std
        # self.P = np.array([[1.0, 0.0], [0.0, 1.0]]) * 100  # Uncertainty covariance.  Large initial
        self.P = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.Q = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.R = meas_noise_std**2  # State uncertainty.  Measurement noise covariance matrix
        self.H = np.array([[1.0, 0.0]])  # Jacobian of h(x).  Measurement matrix (Only measure position)
        self.S = 0.  # System uncertainty
        self.K = 0.  # Kalman gain
        self.hx = 0.  # Output of observation function h(x)
        self.u_kf = 0.  # Control input
        self.x = np.array([initial_position, initial_velocity])  # Kalman state vector [position, velocity]
        self.y_kf = 0.  # Residual z-hx
        self.x = np.array([[0], [0]])
        self.reset = bool(False)

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix + "KF1x1VarDt:\n"
        s += "  Inputs:\n"
        s += "  reset  = " + self.reset + "\n"
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

    def calculate(self, reset=None, dt=None, in_=None):
        self.reset = reset
        self.dt = dt
        if self.reset:
            self.kf_init(in_)
            return in_
        self.predict(self.dt)
        out = self.update(in_)
        return out

    def kf_init(self, in_=None):
        self.u_kf = in_
        self.Fx = [ [1., self.dt], [0., 1.] ]
        self.H = [1., 0.]
        self.K = [0., 0.]
        self.x = np.array([[self.u_kf], [0]])
        for j in range(self.ROWS):
            for i in range(self.COLS):
                self.P[i][j] = 0.
                self.Q[i][j] = 0.

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

        return float(self.x[0])

    def get_state(self):
        """
        Returns the current estimated state.

        Returns:
            numpy.ndarray: The current state vector [position, velocity].
        """
        return float(self.x[0]), float(self.x[1])
