import bpy
import math
import csv
import os
import json
from mathutils import Vector, Euler, Matrix, Quaternion

# ═══════════════════════════════════════════════════════════════════════════
#  ① CONFIGURATION  — edit these values before running
# ═══════════════════════════════════════════════════════════════════════════

# --- Trajectory ---
SQUARE_SIDE    = 2.0      # metres  — side length of the square flight path
CAMERA_HEIGHT  = 1.5      # metres  — fixed altitude above the ground plane
TOTAL_LAPS     = 2        # how many complete square laps to fly
DRONE_SPEED    = 0.3      # m/s     — realistic slow drone cruising speed

# --- Data rate ---
# The project asks for 1000 Hz data; every 10th frame is the "camera" image.
# Keep SIM_HZ high so pose derivatives give good IMU ground truth.
# For quick tests lower it to 100; for final data use 1000.
SIM_HZ         = 100      # simulated sample rate (Hz) for both cam & IMU GT
# NOTE: for the real submission set SIM_HZ = 1000 and use every 10th frame
#       as camera input, all frames as IMU ground truth.

# --- Camera ---
CAMERA_YAW_DEG = 0.0      # yaw of the drone body (0 = +X forward in image)
IMG_WIDTH      = 640      # pixels
IMG_HEIGHT     = 480      # pixels
FOCAL_MM       = 50       # millimetres — equivalent lens focal length

# --- Output ---
OUTPUT_DIR     = "//../../Data/Generated"   # "//" = relative to the .blend file
RENDER_IMAGES  = True             # False = export poses only (much faster)
IMAGE_PREFIX   = "frame_"        # frames will be frame_0001.png etc.

# ═══════════════════════════════════════════════════════════════════════════
#  ② HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_or_create_camera():
    """Return the scene camera.  If none exists, create one."""
    scene = bpy.context.scene
    if scene.camera:
        print(f"[DroneGen] Using existing camera: '{scene.camera.name}'")
        return scene.camera
    cam_data = bpy.data.cameras.new("DroneCamera")
    cam_obj  = bpy.data.objects.new("DroneCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    print("[DroneGen] Created new camera: 'DroneCamera'")
    return cam_obj


def configure_camera_optics(cam_obj):
    """
    Set focal length so the K matrix is known and fixed.
    The sensor size is set to match the render resolution aspect ratio.
    """
    cam_data = cam_obj.data
    cam_data.lens              = FOCAL_MM          # focal length in mm
    cam_data.sensor_fit        = 'HORIZONTAL'
    cam_data.sensor_width      = 36               # mm  (tiny drone cam sensor)
    cam_data.clip_start        = 0.01
    cam_data.clip_end          = 100.0


def compute_K(cam_obj):
    """
    Return the 3×3 intrinsic matrix K as a nested list.
    Uses Blender's camera data + current render resolution.
    """
    scene  = bpy.context.scene
    cam    = cam_obj.data
    W, H   = scene.render.resolution_x, scene.render.resolution_y
    scale  = scene.render.resolution_percentage / 100.0
    W, H   = int(W * scale), int(H * scale)

    if cam.sensor_fit == 'VERTICAL' or (cam.sensor_fit == 'AUTO' and W < H):
        f_px = (cam.lens / cam.sensor_height) * H
    else:
        f_px = (cam.lens / cam.sensor_width) * W

    cx, cy = W / 2.0, H / 2.0
    K = [
        [f_px,  0.0,  cx],
        [ 0.0, f_px,  cy],
        [ 0.0,  0.0, 1.0],
    ]
    return K, W, H


# ───────────────────────────────────────────────────────────────────────────
#  ③ SQUARE TRAJECTORY BUILDER
# ───────────────────────────────────────────────────────────────────────────

def build_square_waypoints(side, height, n_laps):
    """
    Returns corner waypoints for n_laps of a square centred at the origin.
    The last waypoint closes the loop back to the start.

    Corner order (CCW from bottom-left when viewed from above):
        BL → BR → TR → TL → BL …
    """
    h = side / 2.0
    corners = [
        Vector((-h, -h, height)),   # bottom-left
        Vector(( h, -h, height)),   # bottom-right
        Vector(( h,  h, height)),   # top-right
        Vector((-h,  h, height)),   # top-left
    ]
    waypoints = []
    for _ in range(n_laps):
        waypoints.extend(corners)
    waypoints.append(corners[0])    # close the loop
    return waypoints


def interpolate_positions(waypoints, speed, fps):
    """
    Linear interpolation between waypoints at constant speed.
    Returns a list of Vector positions, one per simulated frame.
    """
    positions = []
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        seg_length = (p1 - p0).length
        if seg_length < 1e-6:
            continue
        n_frames = max(2, int(round(seg_length / speed * fps)))
        for f in range(n_frames):
            t = f / n_frames
            positions.append(p0.lerp(p1, t))
    positions.append(waypoints[-1])   # include the final endpoint
    return positions


# ───────────────────────────────────────────────────────────────────────────
#  ④ KEYFRAME INSERTION
# ───────────────────────────────────────────────────────────────────────────

def setup_keyframes(cam_obj, positions, yaw_rad):
    """
    Insert one keyframe per simulated timestep.

    Camera orientation:
      • In Blender, rotation (0, 0, 0) points the camera straight DOWN
        along world –Z (camera local –Z = world –Z).
      • We apply a yaw around world Z to set the drone heading.
      • Roll and Pitch are kept at 0 (< 45° constraint from the spec).

    If your renders look sideways, try changing the base_rot to
      Euler((math.radians(90), 0.0, yaw_rad), 'XYZ')
    """
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end   = len(positions)
    scene.render.fps  = SIM_HZ

    # Downward-facing rotation:  pitch 0°, roll 0°, yaw = yaw_rad
    base_rot = Euler((0.0, 0.0, yaw_rad), 'XYZ')

    # Clear previous animation data
    cam_obj.animation_data_clear()

    for idx, pos in enumerate(positions):
        frame = idx + 1
        scene.frame_set(frame)
        cam_obj.location       = pos
        cam_obj.rotation_euler = base_rot
        cam_obj.keyframe_insert(data_path="location",       frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Force LINEAR interpolation -> constant speed, no easing artefacts.
    # Some Blender builds expose Action.fcurves differently; skip safely.
    if cam_obj.animation_data and cam_obj.animation_data.action:
        action = cam_obj.animation_data.action
        fcurves = getattr(action, "fcurves", None)
        if fcurves is not None:
            for fc in fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'
        else:
            print("[DroneGen] WARNING: Action has no 'fcurves' attribute; "
                  "skipping interpolation override.")

    print(f"[DroneGen] {len(positions)} keyframes inserted "
          f"({len(positions)/SIM_HZ:.2f} s @ {SIM_HZ} Hz)")


# ───────────────────────────────────────────────────────────────────────────
#  ⑤ POSE EXPORT
# ───────────────────────────────────────────────────────────────────────────

def export_poses(cam_obj, n_frames, output_dir):
    """
    Walk through every simulated frame and write the camera world pose.

    CSV columns:
        frame   – 1-indexed frame number
        tx ty tz – world-frame position (metres)
        qw qx qy qz – orientation as unit quaternion (w-first convention)

    The relative pose between consecutive frames is what the network learns.
    Dead-reckoning these incremental poses gives the full odometry trajectory.
    """
    scene    = bpy.context.scene
    csv_path = os.path.join(output_dir, "poses.csv")

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz'])
        for frame in range(1, n_frames + 1):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            mat  = cam_obj.matrix_world
            loc  = mat.to_translation()
            quat = mat.to_quaternion()          # w x y z
            w.writerow([
                frame,
                f"{loc.x:.8f}",  f"{loc.y:.8f}",  f"{loc.z:.8f}",
                f"{quat.w:.8f}", f"{quat.x:.8f}",
                f"{quat.y:.8f}", f"{quat.z:.8f}",
            ])

    print(f"[DroneGen] Poses  → {csv_path}")


def export_camera_K(cam_obj, output_dir):
    """Save the camera intrinsic matrix K to a human-readable text file."""
    K, W, H = compute_K(cam_obj)
    k_path  = os.path.join(output_dir, "camera_K.txt")
    with open(k_path, 'w') as f:
        f.write(f"# Intrinsic matrix K  (image size: {W} x {H})\n")
        f.write(f"# fx  0   cx\n")
        f.write(f"# 0   fy  cy\n")
        f.write(f"# 0   0   1\n\n")
        for row in K:
            f.write("  ".join(f"{v:12.6f}" for v in row) + "\n")
        f.write(f"\nfx = {K[0][0]:.6f}\n")
        f.write(f"fy = {K[1][1]:.6f}\n")
        f.write(f"cx = {K[0][2]:.6f}\n")
        f.write(f"cy = {K[1][2]:.6f}\n")
    print(f"[DroneGen] Camera K → {k_path}")


def export_trajectory_summary(waypoints, positions, output_dir):
    """Save a human-readable description of the planned trajectory."""
    t_path = os.path.join(output_dir, "trajectory.txt")
    with open(t_path, 'w') as f:
        f.write("=== Drone Square Trajectory Summary ===\n\n")
        f.write(f"  Square side      : {SQUARE_SIDE} m\n")
        f.write(f"  Camera height    : {CAMERA_HEIGHT} m\n")
        f.write(f"  Number of laps   : {TOTAL_LAPS}\n")
        f.write(f"  Drone speed      : {DRONE_SPEED} m/s\n")
        f.write(f"  Sim rate (SIM_HZ): {SIM_HZ} Hz\n")
        f.write(f"  Total frames     : {len(positions)}\n")
        f.write(f"  Total duration   : {len(positions)/SIM_HZ:.2f} s\n\n")
        f.write("Waypoints (x, y, z):\n")
        for i, wp in enumerate(waypoints):
            f.write(f"  [{i:02d}]  ({wp.x:+.3f}, {wp.y:+.3f}, {wp.z:+.3f})\n")
    print(f"[DroneGen] Summary  → {t_path}")


# ───────────────────────────────────────────────────────────────────────────
#  ⑥ RENDER SETTINGS & IMAGE EXPORT
# ───────────────────────────────────────────────────────────────────────────

def setup_render(output_dir):
    """
    Configure Blender's render pipeline for fast, consistent output.
    Uses EEVEE (material-preview equivalent) as the spec allows skipping
    photorealistic lighting since sim2real transfer is not required.
    """
    scene = bpy.context.scene
    rd    = scene.render

    # Engine: EEVEE for speed (spec says material-preview is fine)
    scene.render.engine = 'BLENDER_EEVEE'

    # Resolution
    rd.resolution_x          = IMG_WIDTH
    rd.resolution_y          = IMG_HEIGHT
    rd.resolution_percentage = 100

    # Output format
    rd.image_settings.file_format        = 'PNG'
    rd.image_settings.color_mode        = 'RGB'
    rd.image_settings.color_depth       = '8'
    rd.filepath                          = os.path.join(
                                               output_dir, "frames", IMAGE_PREFIX)
    rd.use_file_extension                = True
    rd.use_render_cache                  = False

    # Colour management — keep it neutral / consistent
    scene.view_settings.view_transform  = 'Standard'
    scene.view_settings.look            = 'None'

    # Frame rate must match SIM_HZ so frame numbers are meaningful
    rd.fps                               = SIM_HZ


# ───────────────────────────────────────────────────────────────────────────
#  ⑦ MAIN
# ───────────────────────────────────────────────────────────────────────────

def main():
    # Resolve output path (// = directory of the .blend file)
    output_dir = bpy.path.abspath(OUTPUT_DIR)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("\n" + "═" * 60)
    print("  DroneGen  –  Phase 2 VIO Data Generation")
    print("═" * 60)

    # ── Camera ──────────────────────────────────────────────
    cam = get_or_create_camera()
    configure_camera_optics(cam)

    # ── Trajectory ──────────────────────────────────────────
    waypoints = build_square_waypoints(SQUARE_SIDE, CAMERA_HEIGHT, TOTAL_LAPS)
    positions = interpolate_positions(waypoints, DRONE_SPEED, SIM_HZ)

    print(f"[DroneGen] Square {SQUARE_SIDE}m × {SQUARE_SIDE}m  |  "
          f"{TOTAL_LAPS} lap(s)  |  {len(positions)} frames  |  "
          f"{len(positions)/SIM_HZ:.1f} s")

    # ── Keyframes ───────────────────────────────────────────
    setup_keyframes(cam, positions, math.radians(CAMERA_YAW_DEG))

    # ── Export meta-data ─────────────────────────────────────
    export_poses(cam, len(positions), output_dir)
    export_camera_K(cam, output_dir)
    export_trajectory_summary(waypoints, positions, output_dir)

    # ── Render ───────────────────────────────────────────────
    if RENDER_IMAGES:
        setup_render(output_dir)
        print(f"[DroneGen] Rendering {len(positions)} frames → {frames_dir}")
        print("[DroneGen] (this may take a while — grab a coffee ☕)")
        bpy.ops.render.render(animation=True)
        print("[DroneGen] Render complete ✓")
    else:
        print("[DroneGen] RENDER_IMAGES=False → skipping image render.")

    print("\n[DroneGen] ✅  All done!  Output directory:")
    print(f"           {output_dir}\n")

    # Reminder about the 1-in-10 camera subsampling the spec requires
    print("─" * 60)
    print("  REMINDER (Phase 2 spec):")
    print("  Use every 10th frame as camera input.")
    print(f"  At SIM_HZ={SIM_HZ}, camera effective rate = {SIM_HZ//10} Hz.")
    print("  All frames' poses feed your IMU ground-truth simulator.")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()