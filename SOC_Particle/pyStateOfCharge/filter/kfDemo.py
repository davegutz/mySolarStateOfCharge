import numpy as np
import matplotlib.pyplot as plt
from plot.plq import plq as plq

N = 100


class Saved:
    # For plot savings.   A better way is 'Saver' class in pyfilter helpers and requires making a __dict__
    def __init__(self):
        self.time = []
        self.dt = []
        self.pos = []
        self.velo = []


class KalmanFilter1DVelocity:
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

        # State vector [position, velocity]
        self.x = np.array([[initial_position], [initial_velocity]])

        # Covariance matrix
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]]) * 100  # Large initial uncertainty

        # State transition matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])

        # Measurement matrix (we only measure position)
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance matrix (assuming noise in acceleration)
        self.Q = np.array([
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3, dt**2]
        ]) * proc_noise_std**2

        # Measurement noise covariance matrix
        self.R = np.array([[meas_noise_std**2]])

    def predict(self):
        """
        Performs the prediction step of the Kalman filter.
        Updates the state estimate and covariance matrix based on the model.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        Performs the update step of the Kalman filter.
        Updates the state estimate and covariance matrix based on the new measurement.

        Args:
            measurement (float): The new position measurement.
        """
        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        y = measurement - (self.H @ self.x)  # Innovation
        self.x = self.x + (K @ y)

        # Update covariance matrix
        self.P = (np.eye(self.x.shape[0]) - K @ self.H) @ self.P

    def get_state(self):
        """
        Returns the current estimated state (position and velocity).

        Returns:
            tuple: A tuple containing the estimated position and velocity.
        """
        return self.x[0, 0], self.x[1, 0]


class KalmanFilter1dVarDt:
    """
    A one-dimensional Kalman filter for a signal with process noise.
    Allows for varying update times between predict and update steps.
    """

    def __init__(self, initial_position, initial_velocity, proc_noise_std, meas_noise_std):
        """
        Initializes the Kalman filter.

        Args:
            initial_position (float): The initial estimated state (e.g., position).
            initial_covariance (float): The initial uncertainty in the state estimate.
            proc_noise_std (float): Standard deviation of the process noise (e.g., acceleration noise).
            meas_noise_std (float): Standard deviation of the measurement noise.
        """
        self.x = np.array([initial_position, initial_velocity])  # State vector: [position, velocity]
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]]) * 100  # Large initial uncertainty
        # self.P = np.diag([initial_covariance, 1.0]) # Covariance matrix: [[pos_cov, pos_vel_cov], [vel_pos_cov, vel_cov]]
        self.Q_std = proc_noise_std
        self.R = meas_noise_std**2  # Measurement noise covariance (scalar)

        # Measurement matrix (maps state to measurement: only position is measured)
        self.H = np.array([[1.0, 0.0]])

    def predict(self, dt):
        """
        Performs the prediction step of the Kalman filter.

        Args:
            dt (float): The time difference since the last prediction/update.
        """
        # State transition matrix (constant velocity model)
        F = np.array([[1.0, dt],
                      [0.0, 1.0]])

        # Process noise covariance matrix (assuming noise affects acceleration)
        G = np.array([[0.5 * dt**2],
                      [dt]])
        Q = G @ G.T * self.Q_std**2

        # Predict state and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement):
        """
        Performs the update step of the Kalman filter.

        Args:
            measurement (float): The current measurement.
        """
        # Calculate Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        y = measurement - (self.H @ self.x)  # Innovation
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

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
    dt = 0.1  # Time step (seconds)
    process_noise_std = 0.1  # Standard deviation of acceleration noise
    measurement_noise_std = 0.5  # Standard deviation of position measurement noise
    kf1 = KalmanFilter1DVelocity(initial_position=0.0, initial_velocity=0.0,
                               dt=dt, proc_noise_std=process_noise_std,
                               meas_noise_std=measurement_noise_std)
    kf2 = KalmanFilter1DVelocity(initial_position=0.0, initial_velocity=0.0,
                               dt=dt, proc_noise_std=process_noise_std,
                               meas_noise_std=measurement_noise_std)
    kf3 = KalmanFilter1dVarDt(initial_position=0.0, initial_velocity=0.0,
                              proc_noise_std=process_noise_std,
                              meas_noise_std=measurement_noise_std)
    kf4 = KalmanFilter1dVarDt(initial_position=0.0, initial_velocity=0.0,
                              proc_noise_std=process_noise_std,
                              meas_noise_std=measurement_noise_std)
    mr1 = Saved()
    mv1 = Saved()
    mr2 = Saved()
    mv2 = Saved()
    mr3 = Saved()
    mv3 = Saved()
    mr4 = Saved()
    mv4 = Saved()

    # Simulate some measurements
    true_position1 = 0.0
    true_velocity1 = 1.0  # Constant velocity
    true_position2 = 0.0
    true_velocity2 = 0.0  # Constant velocity
    true_position3 = 0.0
    true_velocity3 = 1.0  # Constant velocity
    true_position4 = 0.0
    true_velocity4 = 0.0  # Constant velocity
    t = -dt
    for i in range(N):
        t += dt
        true_position1 += true_velocity1 * dt
        # Add some random noise to the measurement
        noise = np.random.normal(0, measurement_noise_std)
        noisy_measurement1 = true_position1 + noise
        noisy_measurement2 = true_position2 + noise
        mr1.pos.append(noisy_measurement1)
        mr1.time.append(t)
        mr1.dt.append(dt)
        mr1.velo.append(true_velocity1)
        mr2.pos.append(noisy_measurement2)
        mr2.time.append(t)
        mr2.dt.append(dt)
        mr2.velo.append(true_velocity2)
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

    for x in mr1.pos:
        kf1.predict()
        kf1.update(x)
        pos, vel = kf1.get_state()
        mv1.pos.append(pos)
        mv1.velo.append(vel)
    mv1.time = mr1.time
    mv1.dt = mr1.dt
    for x in mr2.pos:
        kf2.predict()
        kf2.update(x)
        pos, vel = kf2.get_state()
        mv2.pos.append(pos)
        mv2.velo.append(vel)
    mv2.time = mr1.time
    mv2.dt = mr2.dt
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

    run_str1 = 'data 1'
    ver_str1 = 'filtered 1'
    run_str2 = 'data 2'
    ver_str2 = 'filtered 2'
    run_str3 = 'data 1 var dt'
    ver_str3 = 'filtered 1 var dt'
    run_str4 = 'data 2 var dt'
    ver_str4 = 'filtered 2 var dt'

    plt.figure()
    plt.subplot(121)
    plt.title(' kfDemo.py cons dt=0.1')
    plq(plt, mr1, 'time', mr1, 'pos', color='blue', linestyle='-', label='pos1' + run_str1)
    plq(plt, mv1, 'time', mv1, 'pos', color='red', linestyle='--', label='pos1' + ver_str1)
    plq(plt, mr2, 'time', mr2, 'pos', color='magenta', linestyle='-', label='pos2' + run_str2)
    plq(plt, mv2, 'time', mv2, 'pos', color='black', linestyle='--', label='pos2' + ver_str2)
    plt.legend(loc=1)
    plt.subplot(122)
    plq(plt, mr1, 'time', mr1, 'velo', color='blue', linestyle='-', label='velo' + run_str1)
    plq(plt, mv1, 'time', mv1, 'velo', color='red', linestyle='--', label='velo' + ver_str1)
    plq(plt, mr2, 'time', mr2, 'velo', color='magenta', linestyle='-', label='velo' + run_str2)
    plq(plt, mv2, 'time', mv2, 'velo', color='black', linestyle='--', label='velo' + ver_str2)
    plt.legend(loc=1)

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
