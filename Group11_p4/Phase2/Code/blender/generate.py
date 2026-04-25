import bpy
import math
import csv
import os
import json
from datetime import datetime, timezone
from mathutils import Vector, Euler, Matrix, Quaternion

# ═══════════════════════════════════════════════════════════════════════════
#  ① CONFIGURATION  — edit these values before running
# ═══════════════════════════════════════════════════════════════════════════

# --- Trajectory ---
# Shape selector. Add future shapes by extending SHAPE_PARAMS and
# build_path_waypoints().
TRAJECTORY_SHAPE = os.environ.get("DRONE_TRAJECTORY_SHAPE", "square")
# one of: "square", "figure8", "circle"

# Common motion settings used by every trajectory shape.
COMMON_TRAJECTORY_CFG = {
    "height": float(os.environ.get("DRONE_CAMERA_HEIGHT", "1.5")),
    # meters  — fixed altitude above the ground plane
    "laps": 1,       # number of complete closed loops to fly
    "speed": 0.5,    # m/s     — realistic slow drone cruising speed
}

# Shape-specific parameters. Keep each shape payload focused and explicit.
SHAPE_PARAMS = {
    "square": {
        "side": 2.0,  # meters
    },
    "figure8": {
        "width": 3.0,           # meters (span along X)
        "length": 2.0,          # meters (span along Y)
        "samples_per_lap": 400, # control-point density before interpolation
    },
    "circle": {
        "radius": 1.0,          # meters
        "samples_per_lap": 360, # control-point density before interpolation
    },
}

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

ALLOWED_SPLITS = {"train", "val", "test"}
ALLOWED_TEXTURE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")

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


def validate_trajectory_config():
    """
    Validate and normalize trajectory configuration.
    Returns (shape_name, common_cfg, shape_cfg) with clean value types.
    """
    allowed_shapes = {"square", "figure8", "circle"}

    shape_name = str(TRAJECTORY_SHAPE).strip().lower()
    if shape_name not in allowed_shapes:
        raise ValueError(
            f"TRAJECTORY_SHAPE='{TRAJECTORY_SHAPE}' is invalid. "
            f"Choose one of {sorted(allowed_shapes)}"
        )

    required_common = {"height", "laps", "speed"}
    missing_common = sorted(required_common.difference(COMMON_TRAJECTORY_CFG.keys()))
    if missing_common:
        raise ValueError(
            f"COMMON_TRAJECTORY_CFG is missing required keys: {missing_common}"
        )

    common_cfg = {
        "height": float(COMMON_TRAJECTORY_CFG["height"]),
        "laps": int(COMMON_TRAJECTORY_CFG["laps"]),
        "speed": float(COMMON_TRAJECTORY_CFG["speed"]),
    }
    if common_cfg["height"] <= 0.0:
        raise ValueError("COMMON_TRAJECTORY_CFG['height'] must be > 0")
    if common_cfg["laps"] < 1:
        raise ValueError("COMMON_TRAJECTORY_CFG['laps'] must be >= 1")
    if common_cfg["speed"] <= 0.0:
        raise ValueError("COMMON_TRAJECTORY_CFG['speed'] must be > 0")
    if SIM_HZ <= 0:
        raise ValueError("SIM_HZ must be > 0")

    if shape_name not in SHAPE_PARAMS:
        raise ValueError(f"Missing SHAPE_PARAMS entry for shape '{shape_name}'")
    raw_shape_cfg = SHAPE_PARAMS[shape_name]
    if not isinstance(raw_shape_cfg, dict):
        raise ValueError(f"SHAPE_PARAMS['{shape_name}'] must be a dict")

    if shape_name == "square":
        missing = sorted({"side"}.difference(raw_shape_cfg.keys()))
        if missing:
            raise ValueError(f"square config missing required keys: {missing}")
        shape_cfg = {"side": float(raw_shape_cfg["side"])}
        if shape_cfg["side"] <= 0.0:
            raise ValueError("SHAPE_PARAMS['square']['side'] must be > 0")
    elif shape_name == "figure8":
        missing = sorted({"width", "length"}.difference(raw_shape_cfg.keys()))
        if missing:
            raise ValueError(f"figure8 config missing required keys: {missing}")
        shape_cfg = {
            "width": float(raw_shape_cfg["width"]),
            "length": float(raw_shape_cfg["length"]),
            "samples_per_lap": int(raw_shape_cfg.get("samples_per_lap", 400)),
        }
        if shape_cfg["width"] <= 0.0 or shape_cfg["length"] <= 0.0:
            raise ValueError("figure8 width/length must both be > 0")
        if shape_cfg["samples_per_lap"] < 8:
            raise ValueError("figure8 samples_per_lap must be >= 8")
    elif shape_name == "circle":
        missing = sorted({"radius"}.difference(raw_shape_cfg.keys()))
        if missing:
            raise ValueError(f"circle config missing required keys: {missing}")
        shape_cfg = {
            "radius": float(raw_shape_cfg["radius"]),
            "samples_per_lap": int(raw_shape_cfg.get("samples_per_lap", 360)),
        }
        if shape_cfg["radius"] <= 0.0:
            raise ValueError("circle radius must be > 0")
        if shape_cfg["samples_per_lap"] < 8:
            raise ValueError("circle samples_per_lap must be >= 8")
    else:
        raise ValueError(f"Unsupported shape '{shape_name}'")

    return shape_name, common_cfg, shape_cfg


def trajectory_descriptor(shape_name, common_cfg, shape_cfg):
    """Return human-readable shape summary and flat metadata for exports."""
    meta = {
        "shape": shape_name,
        "height": common_cfg["height"],
        "laps": common_cfg["laps"],
        "speed": common_cfg["speed"],
        "sim_hz": SIM_HZ,
    }

    if shape_name == "square":
        meta["side"] = shape_cfg["side"]
        summary = f"square side={shape_cfg['side']:.2f} m"
    elif shape_name == "figure8":
        meta["width"] = shape_cfg["width"]
        meta["length"] = shape_cfg["length"]
        meta["samples_per_lap"] = shape_cfg["samples_per_lap"]
        summary = (
            f"figure8 width={shape_cfg['width']:.2f} m, "
            f"length={shape_cfg['length']:.2f} m"
        )
    elif shape_name == "circle":
        meta["radius"] = shape_cfg["radius"]
        meta["samples_per_lap"] = shape_cfg["samples_per_lap"]
        summary = f"circle radius={shape_cfg['radius']:.2f} m"
    else:
        summary = shape_name

    return summary, meta


def validate_dataset_split(split_name):
    """Validate split label and normalize to lowercase."""
    split = str(split_name).strip().lower()
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"DRONE_DATASET_SPLIT='{split_name}' is invalid. "
            f"Choose one of {sorted(ALLOWED_SPLITS)}"
        )
    return split


def infer_texture_name():
    """Resolve texture/domain tag from env first, then active .blend filename."""
    texture_from_env = os.environ.get("DRONE_TEXTURE_NAME", "").strip()
    if texture_from_env:
        return texture_from_env

    blend_path = bpy.data.filepath
    if blend_path:
        return os.path.splitext(os.path.basename(blend_path))[0]
    return "unknown"


def resolve_texture_image_path(texture_name):
    """Resolve runtime texture image from env override or //textures folder."""
    texture_file_env = os.environ.get("DRONE_TEXTURE_FILE", "").strip()
    if texture_file_env:
        candidate = bpy.path.abspath(texture_file_env)
        if os.path.exists(candidate):
            return candidate
        raise ValueError(
            f"DRONE_TEXTURE_FILE points to a missing file: {candidate}"
        )

    textures_dir = bpy.path.abspath("//textures")
    requested = str(texture_name).strip()
    if not requested:
        raise ValueError("Texture name is empty. Set DRONE_TEXTURE_NAME.")

    has_ext = os.path.splitext(requested)[1].lower() in ALLOWED_TEXTURE_EXTS
    candidates = []
    if has_ext:
        candidates.append(os.path.join(textures_dir, requested))
    else:
        for ext in ALLOWED_TEXTURE_EXTS:
            candidates.append(os.path.join(textures_dir, requested + ext))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    available = []
    if os.path.isdir(textures_dir):
        for name in sorted(os.listdir(textures_dir)):
            ext = os.path.splitext(name)[1].lower()
            if ext in ALLOWED_TEXTURE_EXTS:
                available.append(name)
    raise ValueError(
        f"Texture image not found for '{requested}' in {textures_dir}. "
        f"Available textures: {available}"
    )


def apply_texture_image(texture_image_path):
    """Load one image and assign it to all TEX_IMAGE nodes in materials."""
    image = bpy.data.images.load(texture_image_path, check_existing=True)
    assigned = 0
    for mat in bpy.data.materials:
        node_tree = getattr(mat, "node_tree", None)
        if node_tree is None:
            continue
        for node in node_tree.nodes:
            if node.type == "TEX_IMAGE":
                node.image = image
                assigned += 1

    if assigned == 0:
        raise ValueError(
            "No image texture nodes found in the blend file; cannot apply runtime texture."
        )

    print(
        f"[DroneGen] Applied texture image '{os.path.basename(texture_image_path)}' "
        f"to {assigned} image texture node(s)."
    )


def resolve_sequence_output_dir(base_output_dir, split_name, requested_sequence_id):
    """Create split/sequence output directory and return identifiers + paths."""
    split_dir = os.path.join(base_output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    sequence_id = str(requested_sequence_id).strip()
    if not sequence_id:
        sequence_id = f"seq_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    output_dir = os.path.join(split_dir, sequence_id)
    if os.path.exists(output_dir):
        raise ValueError(
            f"Sequence output already exists: {output_dir}. "
            "Use a unique --seq-id or remove the existing folder."
        )

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=False)
    return sequence_id, output_dir, frames_dir


def write_sequence_metadata(output_dir, metadata):
    """Persist sequence metadata for reproducibility and dataset indexing."""
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"[DroneGen] Metadata → {metadata_path}")


def append_dataset_manifest(base_output_dir, row):
    """Append one sequence record to top-level dataset index.csv."""
    manifest_path = os.path.join(base_output_dir, "index.csv")
    fieldnames = [
        "sequence_id", "split", "shape", "texture", "height_m", "speed_mps",
        "laps", "sim_hz", "num_frames", "duration_s", "seed", "rel_path",
        "timestamp_utc",
    ]

    write_header = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[DroneGen] Manifest → {manifest_path}")


# ───────────────────────────────────────────────────────────────────────────
#  ③ TRAJECTORY BUILDERS
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


def _repeat_closed_loop(base_loop, n_laps):
    """Repeat a closed loop polyline for n_laps without duplicate joins."""
    if n_laps < 1:
        return []
    if not base_loop:
        return []

    waypoints = []
    for lap_idx in range(n_laps):
        if lap_idx == 0:
            waypoints.extend(base_loop)
        else:
            waypoints.extend(base_loop[1:])
    return waypoints


def build_figure8_waypoints(width, length, height, n_laps, samples_per_lap):
    """
    Build a centered figure-8 using a deterministic parametric curve:
      x = a * sin(t),  y = b * sin(t) * cos(t),  t in [0, 2π]
    """
    a = width / 2.0
    b = length / 2.0

    base_loop = []
    for i in range(samples_per_lap):
        t = 2.0 * math.pi * (i / samples_per_lap)
        x = a * math.sin(t)
        y = b * math.sin(t) * math.cos(t)
        base_loop.append(Vector((x, y, height)))
    base_loop.append(base_loop[0].copy())

    return _repeat_closed_loop(base_loop, n_laps)


def build_circle_waypoints(radius, height, n_laps, samples_per_lap):
    """Build a centered circular closed loop sampled uniformly in angle."""
    base_loop = []
    for i in range(samples_per_lap):
        t = 2.0 * math.pi * (i / samples_per_lap)
        x = radius * math.cos(t)
        y = radius * math.sin(t)
        base_loop.append(Vector((x, y, height)))
    base_loop.append(base_loop[0].copy())

    return _repeat_closed_loop(base_loop, n_laps)


def build_path_waypoints(shape_name, common_cfg, shape_cfg):
    """
    Shape dispatcher that returns a closed-loop waypoint list.
    Add future shapes by inserting a new branch here and in SHAPE_PARAMS.
    """
    if shape_name == "square":
        return build_square_waypoints(
            shape_cfg["side"],
            common_cfg["height"],
            common_cfg["laps"],
        )
    if shape_name == "figure8":
        return build_figure8_waypoints(
            shape_cfg["width"],
            shape_cfg["length"],
            common_cfg["height"],
            common_cfg["laps"],
            shape_cfg["samples_per_lap"],
        )
    if shape_name == "circle":
        return build_circle_waypoints(
            shape_cfg["radius"],
            common_cfg["height"],
            common_cfg["laps"],
            shape_cfg["samples_per_lap"],
        )
    raise ValueError(f"Unsupported trajectory shape: {shape_name}")


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


def export_trajectory_summary(waypoints, positions, output_dir, meta):
    """Save a human-readable description of the planned trajectory."""
    t_path = os.path.join(output_dir, "trajectory.txt")
    with open(t_path, 'w') as f:
        f.write("=== Drone Trajectory Summary ===\n\n")
        f.write(f"  Shape            : {meta['shape']}\n")
        if meta["shape"] == "square":
            f.write(f"  Square side      : {meta['side']} m\n")
        elif meta["shape"] == "figure8":
            f.write(f"  Figure8 width    : {meta['width']} m\n")
            f.write(f"  Figure8 length   : {meta['length']} m\n")
            f.write(f"  Samples per lap  : {meta['samples_per_lap']}\n")
        elif meta["shape"] == "circle":
            f.write(f"  Circle radius    : {meta['radius']} m\n")
            f.write(f"  Samples per lap  : {meta['samples_per_lap']}\n")
        f.write(f"  Camera height    : {meta['height']} m\n")
        f.write(f"  Number of laps   : {meta['laps']}\n")
        f.write(f"  Drone speed      : {meta['speed']} m/s\n")
        f.write(f"  Sim rate (SIM_HZ): {meta['sim_hz']} Hz\n")
        f.write(f"  Total frames     : {len(positions)}\n")
        f.write(f"  Total duration   : {len(positions)/SIM_HZ:.2f} s\n\n")

        f.write("Effective parameters (JSON):\n")
        f.write(json.dumps(meta, indent=2) + "\n\n")

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
    print("\n" + "═" * 60)
    print("  DroneGen  –  Phase 2 VIO Data Generation")
    print("═" * 60)

    # Validate trajectory and dataset context before creating output directories.
    shape_name, common_cfg, shape_cfg = validate_trajectory_config()
    split_name = validate_dataset_split(os.environ.get("DRONE_DATASET_SPLIT", "train"))
    requested_sequence_id = os.environ.get("DRONE_SEQUENCE_ID", "")
    seed_tag = os.environ.get("DRONE_DATASET_SEED", "").strip()
    texture_name = infer_texture_name()
    texture_image_path = resolve_texture_image_path(texture_name)

    # Resolve output path (// = directory of the .blend file)
    base_output_dir = bpy.path.abspath(OUTPUT_DIR)
    sequence_id, output_dir, frames_dir = resolve_sequence_output_dir(
        base_output_dir,
        split_name,
        requested_sequence_id,
    )

    # ── Camera ──────────────────────────────────────────────
    cam = get_or_create_camera()
    configure_camera_optics(cam)
    apply_texture_image(texture_image_path)

    # ── Trajectory ──────────────────────────────────────────
    waypoints = build_path_waypoints(shape_name, common_cfg, shape_cfg)
    positions = interpolate_positions(waypoints, common_cfg["speed"], SIM_HZ)
    shape_summary, meta = trajectory_descriptor(shape_name, common_cfg, shape_cfg)
    meta["texture"] = texture_name
    meta["texture_image_file"] = os.path.basename(texture_image_path)
    meta["split"] = split_name
    meta["sequence_id"] = sequence_id
    meta["seed"] = seed_tag
    meta["blend_file"] = os.path.basename(bpy.data.filepath) if bpy.data.filepath else "unknown"

    print(
        f"[DroneGen] Shape={meta['shape']} ({shape_summary})  |  "
        f"height={meta['height']:.2f} m  |  laps={meta['laps']}  |  "
        f"speed={meta['speed']:.2f} m/s"
    )
    print(
        f"[DroneGen] Split={split_name}  |  Seq={sequence_id}  |  "
        f"Texture={texture_name}"
    )
    print(
        f"[DroneGen] {len(positions)} frames  |  "
        f"{len(positions)/SIM_HZ:.2f} s @ {SIM_HZ} Hz"
    )

    # ── Keyframes ───────────────────────────────────────────
    setup_keyframes(cam, positions, math.radians(CAMERA_YAW_DEG))

    # ── Export meta-data ─────────────────────────────────────
    export_poses(cam, len(positions), output_dir)
    export_camera_K(cam, output_dir)
    export_trajectory_summary(waypoints, positions, output_dir, meta)

    rel_path = os.path.relpath(output_dir, base_output_dir)
    metadata = {
        "sequence_id": sequence_id,
        "split": split_name,
        "shape": meta["shape"],
        "texture": texture_name,
        "height_m": meta["height"],
        "speed_mps": meta["speed"],
        "laps": meta["laps"],
        "sim_hz": meta["sim_hz"],
        "num_frames": len(positions),
        "duration_s": len(positions) / SIM_HZ,
        "seed": seed_tag,
        "blend_file": meta["blend_file"],
        "output_dir": output_dir,
        "relative_output_dir": rel_path,
        "trajectory_parameters": meta,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    write_sequence_metadata(output_dir, metadata)
    append_dataset_manifest(base_output_dir, {
        "sequence_id": sequence_id,
        "split": split_name,
        "shape": meta["shape"],
        "texture": texture_name,
        "height_m": f"{meta['height']:.6f}",
        "speed_mps": f"{meta['speed']:.6f}",
        "laps": str(meta["laps"]),
        "sim_hz": str(meta["sim_hz"]),
        "num_frames": str(len(positions)),
        "duration_s": f"{len(positions)/SIM_HZ:.6f}",
        "seed": seed_tag,
        "rel_path": rel_path,
        "timestamp_utc": metadata["generated_utc"],
    })

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