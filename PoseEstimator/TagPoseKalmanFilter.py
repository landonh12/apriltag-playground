import numpy as np

class TagPoseKalmanFilter:
    def __init__(self):
        # State vector: [x, y, z, roll, pitch, yaw]
        self.x = np.zeros(6)  # Initial state estimate
        
        # Covariance matrix: initial uncertainty
        self.P = np.eye(6) * 1e-3  # Small initial uncertainty
        
        # State transition matrix (identity for simplicity)
        self.A = np.eye(6)
        
        # Observation matrix (identity for direct observation)
        self.H = np.eye(6)
        
        # Process noise covariance (model uncertainty)
        # Default constant tune value: 1e-4
        self.Q = np.eye(6) * 1e-6  # Tune this value depending on system dynamics
        
        # Measurement noise covariance (sensor uncertainty)
        # Default constant tune value: 1e-2
        self.R = np.eye(6) * 1e-6  # Tune this based on AprilTag measurement noise
        
    def predict(self):
        # Predict the next state
        self.x = self.A @ self.x  # No control input
        self.P = self.A @ self.P @ self.A.T + self.Q  # Update the covariance
        
    def update(self, z):
        # z is the measurement vector [x, y, z, roll, pitch, yaw] from a tag
        y = z - (self.H @ self.x)  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y  # Update state estimate
        self.P = (np.eye(6) - K @ self.H) @ self.P  # Update covariance

    def get_state(self):
        return self.x