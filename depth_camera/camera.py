import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from threading import Lock
import open3d as o3d
import math
from concurrent.futures import ThreadPoolExecutor

class AppState:
    def __init__(self):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

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
        
    def get_rotation_matrix(self):
        with self.theta_mtx:
            theta = self.theta.copy()

        Rx, _ = cv2.Rodrigues(np.array([[theta[0]], [0], [0]], dtype=np.float32))
        Ry, _ = cv2.Rodrigues(np.array([[0], [theta[1]], [0]], dtype=np.float32))
        Rz, _ = cv2.Rodrigues(np.array([[0], [0], [theta[2]]], dtype=np.float32))

        return np.dot(Rz, np.dot(Ry, Rx)).astype(np.float32)
        

class SLAM:
    def __init__(self):
        self.map_points = None
        self.prev_position = None
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))



    def process_camera_data(self, color_image, depth_image, intrinsics, rotation_estimator=None):
        points_3d, colors = self.extract_3d_points(depth_image, color_image, intrinsics)

        if len(points_3d) > 0 and len(colors) > 0:
            position, orientation = self.estimate_camera_pose(points_3d, colors, intrinsics, rotation_estimator)
            self.update_map(points_3d, colors)

    def extract_3d_points(self, depth_image, color_image, intrinsics):
        points_3d = []
        colors = []

        depth_intrinsics = rs.intrinsics()
        depth_intrinsics.width = intrinsics.width
        depth_intrinsics.height = intrinsics.height
        depth_intrinsics.ppx = intrinsics.ppx
        depth_intrinsics.ppy = intrinsics.ppy
        depth_intrinsics.fx = intrinsics.fx
        depth_intrinsics.fy = intrinsics.fy
        depth_intrinsics.model = intrinsics.model
        depth_intrinsics.coeffs = intrinsics.coeffs

        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                depth = depth_image[v, u]
                if depth > 0:
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                    points_3d.append(point_3d)
                    colors.append(color_image[v, u])

        points_3d = np.array(points_3d, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32) / 255.0

        return points_3d, colors

    def estimate_camera_pose(self, points_3d, colors, intrinsics, rotation_estimator=None):
        if len(points_3d) < 4:
            if self.prev_position is None:
                return np.zeros(3), np.eye(3)
            else:
                return self.prev_position, np.eye(3)

        points_3d = points_3d.astype(np.float32)

        if rotation_estimator:
            imu_rotation = rotation_estimator.get_rotation_matrix()
        else:
            imu_rotation = np.eye(3)

        pixels_2d = np.zeros((len(points_3d), 2), dtype=np.float32)
        for i in range(len(points_3d)):
            pixel = rs.rs2_project_point_to_pixel(intrinsics, points_3d[i])
            pixels_2d[i] = pixel

        camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                  [0, intrinsics.fy, intrinsics.ppy],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)

        _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d, pixels_2d, camera_matrix, dist_coeffs, rvec=cv2.Rodrigues(imu_rotation)[0])

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        position = tvec.flatten()
        orientation = rotation_matrix

        self.prev_position = position

        return position, orientation

    def update_map(self, points_3d, colors):
        if len(points_3d) > 0:
            new_points = o3d.geometry.PointCloud()
            new_points.points = o3d.utility.Vector3dVector(points_3d)
            new_points.colors = o3d.utility.Vector3dVector(colors)

            if self.map_points is None:
                self.map_points = o3d.geometry.PointCloud()

            self.map_points += new_points

            if not hasattr(self, 'vis'):
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window()
                self.vis.add_geometry(self.map_points)
            else:
                self.vis.update_geometry(self.map_points)
                self.vis.poll_events()
                self.vis.update_renderer()

            print("Map updated with new points:", len(points_3d))

    def display_map(self):
        print("Displaying map")
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])

    def close_window(self):
        self.vis.destroy_window()

class IMU:
    def __init__(self):
        self.rotation_estimator = RotationEstimator()

    def process_imu_data(self, gyro_data, accel_data, timestamp):
        self.rotation_estimator.process_gyro(gyro_data, timestamp)
        self.rotation_estimator.process_accel(accel_data)

class Camera(IMU):
    def __init__(self, pipeline, decimate=1):
        super().__init__()
        self.pipeline = pipeline
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, decimate)
        self.colorizer = rs.colorizer()
        self.pc = rs.pointcloud()

    def process_frames(self, color=True):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = self.decimate.process(depth_frame)
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        mapped_frame = color_frame if color else depth_frame
        color_source = np.asanyarray(color_frame.get_data()) if color else depth_colormap

        self.pc.map_to(mapped_frame)
        points = self.pc.calculate(depth_frame)

        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        video_profile = rs.video_stream_profile(depth_frame.profile)
        intrinsics = video_profile.get_intrinsics()

        return verts, texcoords, color_source, intrinsics

class Renderer:
    def __init__(self, app_state):
        self.state = app_state
        cv2.namedWindow(self.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)

        self.mouse_cb = self.create_mouse_callback()
        cv2.setMouseCallback(self.state.WIN_NAME, self.mouse_cb, self)

        self.intrinsics = None
        self.out = None

    def create_mouse_callback(self):
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                param.state.mouse_btns[0] = True
            elif event == cv2.EVENT_LBUTTONUP:
                param.state.mouse_btns[0] = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                param.state.mouse_btns[1] = True
            elif event == cv2.EVENT_RBUTTONUP:
                param.state.mouse_btns[1] = False
            elif event == cv2.EVENT_MBUTTONDOWN:
                param.state.mouse_btns[2] = True
            elif event == cv2.EVENT_MBUTTONUP:
                param.state.mouse_btns[2] = False
            elif event == cv2.EVENT_MOUSEMOVE:
                h, w = param.out.shape[:2]
                dx, dy = x - param.state.prev_mouse[0], y - param.state.prev_mouse[1]
                if param.state.mouse_btns[0]:
                    param.state.yaw += float(dx) / w * 2
                    param.state.pitch -= float(dy) / h * 2
                elif param.state.mouse_btns[1]:
                    dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                    param.state.translation -= np.dot(param.state.rotation, dp)
                elif param.state.mouse_btns[2]:
                    dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                    param.state.translation[2] += dz
                    param.state.distance -= dz
            elif event == cv2.EVENT_MOUSEWHEEL:
                dz = math.copysign(0.1, flags)
                param.state.translation[2] += dz
                param.state.distance -= dz
            param.state.prev_mouse = (x, y)
        return mouse_cb
    
    def render(self, out, verts, texcoords, color_source, intrinsics):
        self.out = out  # Store the out array as an instance variable
        out.fill(0)
        self.intrinsics = intrinsics
        self.draw_pointcloud(out, verts, texcoords, color_source, self.intrinsics)
        self.draw_frustum(out)
        self.draw_axes(out, np.array([0, 0, 0]), self.state.rotation, size=0.1)

        if any(self.state.mouse_btns):
            self.draw_axes(out, self.state.pivot, self.state.rotation, thickness=4)

        cv2.imshow(self.state.WIN_NAME, out)

    def view(self, v):
        return np.dot(v - self.state.pivot, self.state.rotation) + self.state.pivot - self.state.translation

    def project(self, v, intrinsics):
        h, w = intrinsics.height, intrinsics.width
        view_aspect = float(h) / w

        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * (w * view_aspect, h) + (w / 2.0, h / 2.0)

        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj

    def line3d(self, out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        p0 = self.project(pt1.reshape(-1, 3), self.intrinsics)[0]
        p1 = self.project(pt2.reshape(-1, 3), self.intrinsics)[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

    def draw_pointcloud(self, out, verts, texcoords, color, intrinsics, painter=True):
        v = self.view(verts)
        if painter:
            s = v[:, 2].argsort()[::-1]
            proj = self.project(v[s], intrinsics)
        else:
            proj = self.project(v, intrinsics)

        if self.state.scale:
            proj *= 0.5 ** self.state.decimate

        h, w = out.shape[:2]
        j, i = proj.astype(np.uint32).T

        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        np.clip(u, 0, ch - 1, out=u)
        np.clip(v, 0, cw - 1, out=v)

        out[i[m], j[m]] = color[u[m], v[m]]

    def draw_frustum(self, out):
        orig = self.view(np.array([0, 0, 0]))
        w, h = self.intrinsics.width, self.intrinsics.height
        color = (0x40, 0x40, 0x40)
        for d in range(1, 6, 2):
            for (x, y) in [(0, 0), (w, 0), (w, h), (0, h)]:
                p = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], d)
                self.line3d(out, orig, self.view(np.array(p)), color)

    def draw_axes(self, out, pos, rotation, size=0.1, thickness=2):
        self.line3d(out, pos, pos + np.dot((size, 0, 0), rotation), (0xff, 0, 0), thickness)
        self.line3d(out, pos, pos + np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        self.line3d(out, pos, pos + np.dot((0, 0, size), rotation), (0, 0, 0xff), thickness)


# Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

pipeline.start(config)
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

app_state = AppState()
camera = Camera(pipeline)
slam = SLAM()

# Start a thread executor
executor = ThreadPoolExecutor(max_workers=1)

def process_frames():
    while True:
        if not app_state.paused:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Process IMU data
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            if accel_frame and gyro_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                camera.process_imu_data(gyro_data, accel_data, frames.get_timestamp())

            if depth_frame and color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Process camera data for SLAM
                slam.process_camera_data(color_image, depth_image, depth_intrinsics, rotation_estimator=camera.rotation_estimator)

            if not slam.vis.poll_events():
                break

            slam.vis.update_geometry(slam.map_points)
            slam.vis.update_renderer()

# Submit frame processing to the thread executor
future = executor.submit(process_frames)

# Main loop
while True:
    # Wait for the process_frames thread to complete
    future.result()

    if slam.map_points is not None:
        slam.vis.add_geometry(slam.map_points)

    if not slam.vis.poll_events():
        break




pipeline.stop()
slam.close_window()
executor.shutdown()