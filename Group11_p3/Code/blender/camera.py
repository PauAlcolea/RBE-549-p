"""
blender/camera.py
=================
Sets up the Blender camera to match the Tesla dashcam intrinsics
extracted from the calibration videos.

The camera is placed at the estimated ego-vehicle position, oriented
to look forward along the road. Since we're rendering an overlay (not
a full 3D scene), the camera stays fixed per frame — only the objects
in the scene change.

Calibration workflow (one-time, done before running the pipeline):
  1. Use OpenCV's calibrateCamera() with the provided calibration video.
  2. Extract: fx, fy, cx, cy (focal lengths + principal point).
  3. Update config.yaml blender.camera section with these values.
"""

def _pixel_aspect_from_intrinsics(fx: float, fy: float) -> tuple[float, float]:
    """
    Encode fx/fy mismatch using Blender's pixel aspect settings.

    Blender's camera lens models one focal length directly; the other axis can
    be matched by adjusting the render pixel aspect ratio.
    """
    pixel_aspect_x = 1.0
    pixel_aspect_y = 1.0

    if fx > fy:
        pixel_aspect_y = fx / fy
    elif fy > fx:
        pixel_aspect_x = fy / fx

    return pixel_aspect_x, pixel_aspect_y


def _view_fac_in_px(
    sensor_fit: str,
    res_x: int,
    res_y: int,
    pixel_aspect_x: float,
    pixel_aspect_y: float,
) -> float:
    """Compute Blender's effective sensor view size in pixels."""
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    if sensor_fit == "VERTICAL":
        return pixel_aspect_ratio * res_y
    return float(res_x)


def setup_camera(cfg: dict):
    """
    Create and configure the scene camera using intrinsics from config.

    Returns the Blender camera object.
    """
    import bpy, math
    # Creates the Camera object

    # remove any existing camera
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    cam_data = bpy.data.cameras.new(name="DashCam")
    cam_obj  = bpy.data.objects.new("DashCam", cam_data)

    # Add the camera to the scene
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj


    res_x = cfg["blender"]["resolution"][0]
    res_y = cfg["blender"]["resolution"][1]
    fx    = cfg["blender"]["camera"]["fx"]
    fy    = cfg["blender"]["camera"]["fy"]
    cx = cfg["blender"]["camera"]["cx"]
    cy = cfg["blender"]["camera"]["cy"]

    scene = bpy.context.scene
    pixel_aspect_x, pixel_aspect_y = _pixel_aspect_from_intrinsics(fx, fy)
    scene.render.pixel_aspect_x = pixel_aspect_x
    scene.render.pixel_aspect_y = pixel_aspect_y

    sensor_w = 36.0
    sensor_h = 24.0
    cam_data.sensor_width = sensor_w
    cam_data.sensor_height = sensor_h

    # Match the effective image plane that Blender sees after pixel-aspect scaling.
    if pixel_aspect_x * res_x >= pixel_aspect_y * res_y:
        cam_data.sensor_fit = "HORIZONTAL"
        sensor_size_mm = sensor_w
    else:
        cam_data.sensor_fit = "VERTICAL"
        sensor_size_mm = sensor_h

    view_fac_px = _view_fac_in_px(
        cam_data.sensor_fit,
        res_x,
        res_y,
        pixel_aspect_x,
        pixel_aspect_y,
    )
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x

    cam_data.lens = fx * sensor_size_mm / view_fac_px

    # Blender's principal-point offsets use a slightly different sign convention
    # than OpenCV, so we convert from image coordinates explicitly.
    cam_data.shift_x = -(cx - (res_x - 1) / 2.0) / view_fac_px
    cam_data.shift_y = ((cy - (res_y - 1) / 2.0) / view_fac_px) * pixel_aspect_ratio
    
    # Place camera at ego-vehicle position
    h = cfg["blender"]["camera"]["height_m"]
    cam_obj.location = (0, 0, h)
    cam_obj.rotation_euler = (math.pi/2, 0, 0)  # looking forward (+Y)
    
    return cam_obj