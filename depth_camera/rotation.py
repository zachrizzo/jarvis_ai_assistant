import math
import numpy as np
import pyrealsense2 as rs
import cv2
from threading import Lock
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class rotation_estimator:
    def __init__(self):
        self.theta = np.array([0, 0, 0], dtype=np.float32)
        self.theta_mtx = Lock()
        self.alpha = 0.98
        self.first = True
        self.last_ts_gyro = 0

    def process_gyro(self, gyro_data, ts):
        if self.first:
            self.last_ts_gyro = ts
            return

        gyro_angle = np.array([gyro_data.x, gyro_data.y, gyro_data.z], dtype=np.float32)

        dt_gyro = (ts - self.last_ts_gyro) / 1000.0
        self.last_ts_gyro = ts

        gyro_angle = gyro_angle * dt_gyro

        with self.theta_mtx:
            self.theta += np.array([-gyro_angle[2], -gyro_angle[1], gyro_angle[0]])

    def process_accel(self, accel_data):
        accel_angle = np.array([0, 0, 0], dtype=np.float32)

        accel_angle[2] = math.atan2(accel_data.y, accel_data.z)
        accel_angle[0] = math.atan2(accel_data.x, math.sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z))

        with self.theta_mtx:
            if self.first:
                self.first = False
                self.theta = accel_angle
                self.theta[1] = math.pi
            else:
                self.theta[0] = self.theta[0] * self.alpha + accel_angle[0] * (1 - self.alpha)
                self.theta[2] = self.theta[2] * self.alpha + accel_angle[2] * (1 - self.alpha)

    def get_theta(self):
        with self.theta_mtx:
            return self.theta.copy()

pipeline = rs.pipeline()
config = rs.config()

# Get the first connected RealSense device
context = rs.context()
device = context.devices[0]

# Enable IMU streams with the specific device
config.enable_device(device.get_info(rs.camera_info.serial_number))
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

algo = rotation_estimator()

def imu_callback(frame):
    motion = frame.as_motion_frame()
    if motion:
        if motion.get_profile().stream_type() == rs.stream.gyro:
            ts = frame.get_timestamp()
            gyro_data = motion.get_motion_data()
            algo.process_gyro(gyro_data, ts)

        if motion.get_profile().stream_type() == rs.stream.accel:
            accel_data = motion.get_motion_data()
            algo.process_accel(accel_data)

pipeline.start(config, callback=imu_callback)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the initial viewing angle
ax.view_init(elev=30, azim=45)

# Create a line object for the x, y, and z axes
line_x, = ax.plot([], [], [], 'r-', label='X')
line_y, = ax.plot([], [], [], 'g-', label='Y')
line_z, = ax.plot([], [], [], 'b-', label='Z')

# Set the plot limits and labels
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.ion()
plt.show()

try:
    while True:
        theta = algo.get_theta()
        print("Rotation: ", theta)

        # Update the line data based on the rotation angles
        line_x.set_data([0, math.cos(theta[2])], [0, math.sin(theta[2])])
        line_x.set_3d_properties([0, 0])

        line_y.set_data([0, 0], [0, math.cos(theta[0])])
        line_y.set_3d_properties([0, math.sin(theta[0])])

        line_z.set_data([0, math.sin(theta[1])], [0, 0])
        line_z.set_3d_properties([0, math.cos(theta[1])])

        # Refresh the plot
        fig.canvas.draw()
        fig.canvas.flush_events()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()