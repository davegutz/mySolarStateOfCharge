import numpy as np
import matplotlib.pyplot as plt
from DataOverModel import plq


class KalmanFilter1DVelocity:
    def __init__(self, initial_position, initial_velocity, dt, process_noise_std, measurement_noise_std):
        """
        Initializes a 1D Kalman filter with a constant velocity model.

        Args:
            initial_position (float): Initial estimate of the position.
            initial_velocity (float): Initial estimate of the velocity.
            dt (float): Time step between measurements.
            process_noise_std (float): Standard deviation of the process noise (acceleration).
            measurement_noise_std (float): Standard deviation of the measurement noise (position).
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
        ]) * process_noise_std**2

        # Measurement noise covariance matrix
        self.R = np.array([[measurement_noise_std**2]])

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

# Example Usage:
if __name__ == "__main__":
    dt = 0.1  # Time step (seconds)
    process_noise_std = 0.1  # Standard deviation of acceleration noise
    measurement_noise_std = 0.5  # Standard deviation of position measurement noise

    kf = KalmanFilter1DVelocity(initial_position=0.0, initial_velocity=0.0,
                               dt=dt, process_noise_std=process_noise_std,
                               measurement_noise_std=measurement_noise_std)

    # Simulate some measurements
    true_position = 0.0
    true_velocity = 1.0  # Constant velocity
    measurements = []
    velo = []
    time = []
    t = -dt
    for i in range(50):
        t += dt
        true_position += true_velocity * dt
        # Add some random noise to the measurement
        noisy_measurement = true_position + np.random.normal(0, measurement_noise_std)
        measurements.append(noisy_measurement)
        time.append(t)
        velo.append(true_velocity)

    estimated_positions = []
    estimated_velocities = []

    for measurement in measurements:
        kf.predict()
        kf.update(measurement)
        pos, vel = kf.get_state()
        estimated_positions.append(pos)
        estimated_velocities.append(vel)

    print("Estimated Positions:", estimated_positions[:5])
    print("Estimated Velocities:", estimated_velocities[:5])

    mrr = []
    mrr.time = time
    mrr.meas = measurements
    mrr.velo = velo
    mr = np.recarray(mrr)
    mvv = []
    mvv.time = time
    mvv.emeas = estimated_positions
    mvv.evelo = estimated_velocities
    mv = np.recarray(mvv)
    run_str = 'data'
    ver_str = 'filtered'

    plt.figure()
    plt.subplot(121)
    plt.title(' kfDemo.py ')
    plq(plt, mr, 'time', mr, 'meas', color='black', linestyle='-', label='meas' + run_str)
    plq(plt, mv, 'time', mv, 'emeas', color='cyan', linestyle='--', label='emeas' + ver_str)
    plt.legend(loc=1)
    plt.subplot(122)
    plq(plt, mr, 'time', mr, 'velo', color='black', linestyle='-', label='velo' + run_str)
    plq(plt, mv, 'time', mv, 'evelo', color='cyan', linestyle='--', label='evelo' + ver_str)
    plt.legend(loc=1)
    plt.show(block=False)
