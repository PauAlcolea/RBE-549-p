"""
blender/scene.py
================
Master Blender script. Invoked headlessly by run_blender.sh:

  blender --background --python blender/scene.py -- --seq Seq1 --config config.yaml

For each frame in the sequence:
  1. Load the per-frame detection JSON
  2. Set up / update the Blender scene
  3. Place 3D assets (cars, pedestrians, stop signs)
  4. Draw lane geometry
  5. Set traffic light material colors
  6. Render to PNG
After all frames: stitch PNGs → MP4.
"""

import sys
import argparse
from pathlib import Path

# sys.path must include Code/ so our utils and blender packages are importable
code_dir = Path(__file__).resolve().parent.parent
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from utils.io_utils import load_detection_json, list_frame_jsons, load_config
from blender.render  import render_frame, frames_to_video
from blender.camera import setup_camera
from blender.assets import AssetLibrary
from blender.lanes import LaneRenderer

# bpy is only available inside Blender — guarded below
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def parse_args():
    """
    Parse arguments passed after the '--' separator in the Blender command.
    sys.argv looks like: ['blender', ..., '--', '--scene', 'scene1', ...]
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",  required=True, help="Sequence name, e.g. scene1")
    parser.add_argument("--cam",    required=True, help="Camera: front | back | left_repeater | right_repeater")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--debug",  action="store_true",   help="Print extra diagnostics per frame")
    return parser.parse_args(argv)


def setup_scene(cfg: dict):
    """
    One-time Blender scene setup — call this once before the frame loop.
 
    - Delete everything Blender puts in a default scene (cube, camera, light)
    - Set render resolution and engine from config
    - Set background to solid black (Tesla-style dark environment)
    - film_transparent = True so we can later composite onto the real frame
    - EEVEE samples not applicable; CYCLES sample count set from config
    """
    # Delete all default objects (cube, camera, light)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
 
    scene = bpy.context.scene
    scene.render.engine       = cfg["blender"]["render_engine"]        # "CYCLES" or "EEVEE"
    scene.render.resolution_x = cfg["blender"]["resolution"][0]        # 1920
    scene.render.resolution_y = cfg["blender"]["resolution"][1]        # 1080

    # make a light
    lamp_data = bpy.data.lights.new(name="TestSun", type="SUN")
    lamp_obj  = bpy.data.objects.new("TestSun", lamp_data)
    bpy.context.collection.objects.link(lamp_obj)
    lamp_obj.location = (5, 5, 10)

    # make ground plane
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.location.z = 0       # slightly below 0 so objects sit on z=0
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)  # dark gray
    ground.data.materials.append(mat)

    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True                               # RGBA output, background = alpha 0
 
    if cfg["blender"]["render_engine"] == "CYCLES":
        scene.cycles.samples = cfg["blender"]["samples"]
 
    # Black world background (visible where film_transparent is off, and for ambient)
    world = bpy.data.worlds.new("EV_World")
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value    = (0.0, 0.0, 0.0, 1.0)   # black
    bg_node.inputs["Strength"].default_value = 0.0                     # no ambient light from world
    scene.world = world
 
    print("[scene] setup_scene complete — "
          f"engine={scene.render.engine}  "
          f"res={scene.render.resolution_x}x{scene.render.resolution_y}")


def render_sequence(scene_name: str, camera: str, cfg: dict, debug: bool = False, asset_lib=None):
    """
    Main loop: iterate over all detection JSONs for one scene+camera pair,
    update the scene per frame, and render each frame to PNG.
    Then stitch all PNGs to an MP4.
    """

    det_dir  = Path(cfg["paths"]["detections_dir"]) / scene_name / camera
    frames_dir = Path(cfg["paths"]["frames_dir"])   / scene_name / camera
    video_dir  = Path(cfg["paths"]["videos_dir"])
 
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
 
    jsons = list_frame_jsons(det_dir)
    if not jsons:
        print(f"[scene] WARNING: no JSONs found in {det_dir} — nothing to render")
        return
 
    print(f"[scene] Rendering {len(jsons)} frames for {scene_name}/{camera}")
    
    # Camera is fixed for the entire sequence, the objects are the ones that move around
    setup_camera(cfg)

    # Lane renderer: persists across frames, geometry cleared per frame
    lane_renderer = LaneRenderer(cfg)

    for i, json_path in enumerate(jsons):
        frame_data = load_detection_json(json_path)
        frame_idx  = frame_data["frame"]
 
        # ── Per-frame work goes here as we implement Steps 3-7 ──────────────
        # Step 3: camera.setup_camera(cfg)          (added in Step 3)
        # Step 4: asset_lib.place_vehicle(v) etc.   (added in Step 4)
        # Step 6: materials                          (added in Step 6)
        # Step 7: traffic lights / stop signs        (added in Step 7)
        # ────────────────────────────────────────────────────────────────────
        
        # empty the frame from all the objects
        if asset_lib:
            asset_lib.clear_frame_objects()

        # clear previous frame's lane geometry
        lane_renderer.clear()

        # go through all of the assets detected and place vehicles and pedestrians
        if asset_lib:
            print("vehicles in frame: ")
            for v in frame_data.get("vehicles", []):
                print(v)
                asset_lib.place_vehicle(v)
            print("pedestrians in frame: ")
            for p in frame_data.get("pedestrians", []):
                print(p)
                asset_lib.place_pedestrian(p)
            # traffic lights
            print("traffic lights in frame: ")
            for t in frame_data.get("traffic_lights", []):
                print(t)
                asset_lib.place_traffic_light(t)

            # lanes
            print("lanes in frame: ")
            for lane in frame_data.get("lanes", []):
                print(lane)
                lane_renderer.draw_lane(lane)

            # some white space for debugging
            print()

 
        out_png = frames_dir / f"frame_{frame_idx:06d}.png"
        print("Objects in scene:", [(o.name, o.type) for o in bpy.data.objects])

        render_frame(cfg, out_png)
 
        if debug:
            print(f"  [{scene_name}/{camera}] frame {frame_idx:06d} → {out_png.name}")
 
        if (i + 1) % 20 == 0:
            print(f"  [{scene_name}/{camera}] {i + 1}/{len(jsons)} frames rendered")
 
    # Stitch all PNGs to MP4
    out_mp4 = video_dir / f"{scene_name}_{camera}.mp4"
    frames_to_video(frames_dir, out_mp4, fps=cfg["blender"]["fps"])
    print(f"[scene] Done → {out_mp4}")


def main():
    args = parse_args()
 
    # config path is relative to Code/ unless absolute
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = code_dir / config_path
 
    cfg = load_config(str(config_path))
 
    setup_scene(cfg)
    asset_lib = AssetLibrary(cfg)
    render_sequence(args.scene, args.cam, cfg, debug=args.debug, asset_lib=asset_lib)
 
 
if __name__ == "__main__":
    if not IN_BLENDER:
        print("ERROR: This script must be run inside Blender:")
        print("  blender --background --python blender/scene.py -- --scene scene1 --cam front")
        sys.exit(1)
    main()