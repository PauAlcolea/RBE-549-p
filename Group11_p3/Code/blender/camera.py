# """
# blender/camera.py
# =================
# Sets up the Blender camera to match the Tesla dashcam intrinsics
# extracted from the calibration videos.

# The camera is placed at the estimated ego-vehicle position, oriented
# to look forward along the road. Since we're rendering an overlay (not
# a full 3D scene), the camera stays fixed per frame — only the objects
# in the scene change.

# Calibration workflow (one-time, done before running the pipeline):
#   1. Use OpenCV's calibrateCamera() with the provided calibration video.
#   2. Extract: fx, fy, cx, cy (focal lengths + principal point).
#   3. Update config.yaml blender.camera section with these values.
# """


# def setup_camera(cfg: dict):
#     """
#     Create and configure the scene camera using intrinsics from config.

#     Returns the Blender camera object.
#     """
#     # TODO: implement
#     # import bpy, math
#     # cam_data = bpy.data.cameras.new(name="DashCam")
#     # cam_obj  = bpy.data.objects.new("DashCam", cam_data)
#     # bpy.context.collection.objects.link(cam_obj)
#     # bpy.context.scene.camera = cam_obj
#     #
#     # res_x = cfg["blender"]["resolution"][0]
#     # res_y = cfg["blender"]["resolution"][1]
#     # fx    = cfg["blender"]["camera"]["fx"]
#     # fy    = cfg["blender"]["camera"]["fy"]
#     #
#     # # Blender uses sensor width + focal_length to compute FOV.
#     # # Easiest approach: use sensor_fit="HORIZONTAL" and compute lens from fx.
#     # sensor_w = 36.0  # mm, standard 35mm equivalent
#     # cam_data.sensor_fit    = "HORIZONTAL"
#     # cam_data.sensor_width  = sensor_w
#     # cam_data.lens = (fx / res_x) * sensor_w
#     #
#     # # Place camera at ego-vehicle position
#     # h = cfg["blender"]["camera"]["height_m"]
#     # cam_obj.location = (0, 0, h)
#     # cam_obj.rotation_euler = (math.pi/2, 0, 0)  # looking forward (+Y)
#     #
#     # return cam_obj
#     raise NotImplementedError("setup_camera not yet implemented")


# def camera_from_calibration_video(video_path: str, out_config_path: str):
#     """
#     Helper (run once outside Blender) to calibrate the camera from the
#     provided calibration video using a checkerboard pattern.

#     Writes the resulting fx, fy, cx, cy into config.yaml.

#     Usage
#     -----
#     python -c "from blender.camera import camera_from_calibration_video; \
#                camera_from_calibration_video('../Data/Calib/calib.mp4', 'config.yaml')"
#     """
#     # TODO: implement
#     # import cv2, yaml, numpy as np
#     # CHECKERBOARD = (9, 6)  # adjust to your actual checkerboard
#     # objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
#     # objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
#     # ... collect corners from video frames ...
#     # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(...)
#     # fx, fy = mtx[0,0], mtx[1,1]
#     # cx, cy = mtx[0,2], mtx[1,2]
#     # ... write to config.yaml ...
#     raise NotImplementedError("camera_from_calibration_video not yet implemented")
