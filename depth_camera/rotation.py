import math
import numpy as np
import pyrealsense2 as rs
import cv2
from threading import Lock

global_color_frame = None
global_depth_frame = None
frames_lock = Lock()

class SLAM:
    def __init__(self):
        self.rotation_estimator = RotationEstimator()
        self.path = []
        self.prev_position = None

    def process_imu_data(self, gyro_data, accel_data, timestamp):
        self.rotation_estimator.process_gyro(gyro_data, timestamp)
        self.rotation_estimator.process_accel(accel_data)

    def update_path(self, position):
        if self.prev_position is None or not np.array_equal(position, self.prev_position):
            self.path.append(position.copy())
            self.prev_position = position.copy()
            print("Path updated: ", position)

    def display_map(self, width, height, focal_length, center_x, center_y, zoom_factor):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i + 1]
            if p1[2] * zoom_factor != 0 and p2[2] * zoom_factor != 0:
                x1, y1 = int(p1[0] * focal_length / (p1[2] * zoom_factor) + center_x), int(p1[1] * focal_length / (p1[2] * zoom_factor) + center_y)
                x2, y2 = int(p2[0] * focal_length / (p2[2] * zoom_factor) + center_x), int(p2[1] * focal_length / (p2[2] * zoom_factor) + center_y)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("3D Space", frame)

class RotationEstimator:
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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
slam = SLAM()

def imu_callback(frame):
    global global_color_frame, global_depth_frame
    motion = frame.as_motion_frame()
    if motion:
        if motion.get_profile().stream_type() == rs.stream.gyro:
            ts = frame.get_timestamp()
            gyro_data = motion.get_motion_data()
            print("Gyro data: ", gyro_data)
            slam.process_imu_data(gyro_data, None, ts)

        if motion.get_profile().stream_type() == rs.stream.accel:
            accel_data = motion.get_motion_data()
            print("Accel data: ", accel_data)
            slam.process_imu_data(None, accel_data, None)
    if frame.is_frameset():
        frames = frame.as_frameset()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame and depth_frame:
            with frames_lock:
                global_color_frame = np.asanyarray(color_frame.get_data())
                global_depth_frame = np.asanyarray(depth_frame.get_data())

pipeline.start(config, callback=imu_callback)

width, height = 800, 600
fov = 60
focal_length = width / (2 * math.tan(fov / 2 * math.pi / 180))
center_x, center_y = width // 2, height // 2

cv2.namedWindow("3D Space", cv2.WINDOW_NORMAL)
cv2.resizeWindow("3D Space", width, height)

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
    while True:
        with frames_lock:
            # Directly use the global variables for the current iteration of the loop
            color_image = global_color_frame.copy() if global_color_frame is not None else None
            depth_image = global_depth_frame.copy() if global_depth_frame is not None else None

        if color_image is not None and depth_image is not None:
            # The images are already numpy arrays; no need for further conversion

            # You would need to ensure `process_camera_data` handles these numpy arrays correctly
            slam.process_camera_data(color_image, depth_image)

            theta = slam.rotation_estimator.get_theta()
            print("Rotation: ", theta)

            sensitivity = 0.5
            theta *= sensitivity

            rotation_matrix_x = cv2.Rodrigues(np.array([theta[0], 0, 0]))[0]
            rotation_matrix_y = cv2.Rodrigues(np.array([0, theta[1], 0]))[0]
            rotation_matrix_z = cv2.Rodrigues(np.array([0, 0, theta[2]]))[0]
            orientation = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

            zoom_factor += zoom_speed * (1 if delta_y > 0 else -1)
            zoom_factor = max(0.1, min(5.0, zoom_factor))

            slam.display_map(width, height, focal_length, center_x, center_y, zoom_factor)

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