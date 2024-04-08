import math
import numpy as np
import pyrealsense2 as rs
import cv2
from threading import Lock

class rotation_estimator:
    def __init__(self):
        self.theta = np.array([0, 0, 0], dtype=np.float32)
        self.theta_mtx = Lock()
        self.alpha = 0.98
        self.first = True
        self.last_ts_gyro = 0
        self.accel_data = None

    def process_gyro(self, gyro_data, ts):
        if self.first:
            self.last_ts_gyro = ts
            self.first = False
            return

        dt_gyro = (ts - self.last_ts_gyro) / 1000.0
        self.last_ts_gyro = ts

        gyro_angle = np.array([gyro_data.x, gyro_data.y, gyro_data.z], dtype=np.float32) * dt_gyro

        with self.theta_mtx:
            self.theta += gyro_angle

    def process_accel(self, accel_data):
        self.accel_data = accel_data

        # Define a threshold for movement detection
        move_threshold = 5  # Adjust based on experimentation

        accel_angle = np.array([0, 0, 0], dtype=np.float32)
        accel_magnitude = np.sqrt(accel_data.x**2 + accel_data.y**2 + accel_data.z**2)
        accel_data_normalized = np.array([accel_data.x / accel_magnitude, accel_data.y / accel_magnitude, accel_data.z / accel_magnitude])

        if np.abs(accel_data_normalized[0]) > move_threshold or np.abs(accel_data_normalized[1]) > move_threshold:
            accel_angle[0] = math.atan2(accel_data.x, math.sqrt(accel_data.y**2 + accel_data.z**2))
            accel_angle[1] = math.atan2(accel_data.y, math.sqrt(accel_data.x**2 + accel_data.z**2))
            accel_angle[2] = math.atan2(math.sqrt(accel_data.x**2 + accel_data.y**2), accel_data.z)

        with self.theta_mtx:
            self.theta = self.theta * self.alpha + accel_angle * (1 - self.alpha)

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
            print("Gyro data: ", gyro_data)
            algo.process_gyro(gyro_data, ts)

        if motion.get_profile().stream_type() == rs.stream.accel:
            accel_data = motion.get_motion_data()
            print("Accel data: ", accel_data)
            algo.process_accel(accel_data)

pipeline.start(config, callback=imu_callback)

width, height = 800, 600
fov = 60
focal_length = width / (2 * math.tan(fov / 2 * math.pi / 180))
center_x, center_y = width // 2, height // 2

cv2.namedWindow("3D Space", cv2.WINDOW_NORMAL)
cv2.resizeWindow("3D Space", width, height)

position = np.array([0, 0, 0], dtype=np.float32)
orientation = np.eye(3, dtype=np.float32)
path = []

mouse_down = False
prev_mouse_x, prev_mouse_y = 0, 0
delta_x, delta_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global mouse_down, prev_mouse_x, prev_mouse_y, delta_x, delta_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
        prev_mouse_x, prev_mouse_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_down:
            delta_x = x - prev_mouse_x
            delta_y = y - prev_mouse_y
            prev_mouse_x, prev_mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        delta_x, delta_y = 0, 0

cv2.setMouseCallback("3D Space", mouse_callback)

zoom_factor = 1.0
zoom_speed = 0.1

try:
    prev_position = None
    while True:
        theta = algo.get_theta()
        print("Rotation: ", theta)

        sensitivity = 0.5
        theta *= sensitivity

        rotation_matrix_x = cv2.Rodrigues(np.array([theta[0], 0, 0]))[0]
        rotation_matrix_y = cv2.Rodrigues(np.array([0, theta[1], 0]))[0]
        rotation_matrix_z = cv2.Rodrigues(np.array([0, 0, theta[2]]))[0]
        orientation = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

        accel_data = algo.accel_data
        if accel_data is not None:
            acceleration = np.array([accel_data.x, accel_data.y, accel_data.z], dtype=np.float32)
            acceleration_magnitude = np.linalg.norm(acceleration)
            if acceleration_magnitude > 0.1:
                velocity = np.dot(orientation, acceleration)
                position += velocity * 0.01

        if prev_position is None or not np.array_equal(position, prev_position):
            path.append(position.copy())
            prev_position = position.copy()
            print("Path updated: ", position)

        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            if p1[2] * zoom_factor != 0 and p2[2] * zoom_factor != 0:
                x1, y1 = int(p1[0] * focal_length / (p1[2] * zoom_factor) + center_x), int(p1[1] * focal_length / (p1[2] * zoom_factor) + center_y)
                x2, y2 = int(p2[0] * focal_length / (p2[2] * zoom_factor) + center_x), int(p2[1] * focal_length / (p2[2] * zoom_factor) + center_y)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        zoom_factor += zoom_speed * (1 if delta_y > 0 else -1)
        zoom_factor = max(0.1, min(5.0, zoom_factor))

        cv2.imshow("3D Space", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            zoom_factor += 0.1
        elif key == ord('-') or key == ord('_'):
            zoom_factor -= 0.1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
