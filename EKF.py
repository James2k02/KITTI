import numpy as np
import cv2
from filterpy.kalman import ExtendedKalmanFilter as EKF
from scipy.spatial.transform import Rotation as R

class EKF_VIO:
    def __init__(self, dt = 0.01):        
        # dt: time interval between two frames
        self.dt = dt
        
        # State vector: [x, y, z, vx, vy, vz, qx, qy, qz, qw, bax, bay, baz, bgx, bgy, bgz]
        # there are 16 states in total
        self.x = np.zeros(16) # a NumPy array of 16 zeros representing each state variable
        self.x[9] = 1 # initial orientation quaternion (no rotation)
        
        # EKF initialization
        self.ekf = EKF(dim_x = 16, dim_z = 6) # state vector and measurement vector have 16 and 6 (x, y, z, roll, pitch, yaw) dimensions respectively
        self.ekf.x = self.x # initial state vector
        self.ekf.P = np.eye(16) * 0.1 # initial state covariance matrix (uncertainty); can change initial value to modify initial uncertainty
        
        # Process noise covariance matrix
        self.ekf.Q = np.eye(16) * 0.001 
        
        # Measurement noise covariance matrix
        self.ekf.R = np.eye(6) * 0.05
        
        # Measurement matrix
        self.ekf.H = np.zeros((6, 16))
        self.ekf.H[:3, :3] = np.eye(3) # position measurement
        self.ekf.H[3:, 6:10] = np.eye(4)[:3] # orientation measurement
        
    def state_transition(self, x, u):
        dt = self.dt
        ax, ay, az, gx, gy, gz = u # acceleration and gyroscope measurements
        
        # Transform accleration from body to NED frame
        R_NB = R.from_quat(x[6:10]).as_matrix() # rotation matrix from body to NED frame
        accel_NED = R_NB @ np.array([ax, ay, az]) # acceleration in NED frame
        
        # Position and Velocity State Transition
        pos = x[:3] + x[3:6] * dt + 0.5 * accel_NED * dt**2
        vel = x[3:6] + accel_NED * dt
        
        # Quaternion State Transition
        q = x[6:10]
        omega = np.array([0, gx, gy, gz])
        q_dot = 0.5 * self.quaternion_product(q, omega)
        quat = q + q_dot * dt
        quat /= np.linalg.norm(quat)
        
        # Biases are kept constant
        biases = x[10:]
        
        return np.concatenate([pos, vel, quat, biases]) # merges into a single matrix (vertically)
    
    def quaternion_product(self, q, omega):
        """ This essentially calculates the skew matrix and multiplies it with the quaternion at the same time instead of 2 separate steps """
        q0, q1, q2, q3 = q
        w0, w1, w2, w3 = omega
        return np.array([
            -q1*w1 - q2*w2 - q3*w3,
            q0*w1 + q2*w3 - q3*w2,
            q0*w2 - q1*w3 + q3*w1,
            q0*w3 + q1*w2 - q2*w1
        ])
        
    def measurement_func(self, x): # this transforms the state vector x into the measurement space z
        return np.concatenate([x[:3], self.quaternion_to_euler(x[6:10])]) # first 3 elements are position and the rest are quaternions converted to euler angles
    # this function defines what the EKF "sees"; it maps the full state to only the observable parts (position and attitude)
    # links the EKF prediction to the measurement --> used in the update step to correct the state
    
    def predict(self, accel, gyro): # accel and gyro are the IMU measurements        
        """ Predicts the next state using only the IMU data (accelerometer and gyroscope) """
        u = np.concatenate([accel - self.ekf.x[10:13], gyro - self.ekf.x[13:16]]) # subtracts the biases from the IMU data to remove long-term draft from sensor readings
        self.ekf.F = self.ekf.compute_jacobian_F(self.ekf.x, u) # computes the Jacobian of the state transition function
        self.ekf.predict(u = u, dt = self.dt, fx = self.state_transition) # updates the state vector using the state transition function
        
    def update(self, z): # z is the measurement vector
         """ Updates the state using the VO measurements"""
         self.ekf.update(z, HJacobian = self.jacobian_H, Hx = self.measurement_func)
         
    def compute_jacobian_F(self, x, u): 
        """ Computes the full process Jacobian matrix F """
        dt = self.dt
        F = np.eye(16)
        F[:3, 3:6] = np.eye(3) * dt # adds a relationship between position and velocity
        
        # Compute the dv/dq term
        a_B = np.array([u[0], u[1], u[2]]) # acceleration in body frame
        Q_F = np.array([
            [x[6], x[9], -x[8], x[7]],
            [x[7], x[8], x[9], -x[6]],
            [x[8], -x[7], x[6], x[9]]
        ])
        dv_dq = 2* Q_F @ np.vstack((np.zeros((1, 3)), a_B.reshape(3, 1).T))
        F[3:6, 6:10] = dv_dq * dt # adds a relationship between velocity and quaternion
        
        # Quaternion update Jacobian
        omega = np.array([
            [0, -u[3], -u[4], -u[5]],
            [u[3], 0, u[5], -u[4]],
            [u[4], -u[5], 0, u[3]],
            [u[5], u[4], -u[3], 0]
        ])
        F[6:10, 6:10] += 0.5 * omega * dt  # quaternion evolution
        
        return F
        
        
        
        