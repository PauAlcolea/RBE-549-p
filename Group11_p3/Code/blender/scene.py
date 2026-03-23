# """
# blender/scene.py
# ================
# Master Blender script. Invoked headlessly by run_blender.sh:

#   blender --background --python blender/scene.py -- --seq Seq1 --config config.yaml

# For each frame in the sequence:
#   1. Load the per-frame detection JSON
#   2. Set up / update the Blender scene
#   3. Place 3D assets (cars, pedestrians, stop signs)
#   4. Draw lane geometry
#   5. Set traffic light material colors
#   6. Render to PNG
# After all frames: stitch PNGs → MP4.
# """

# import sys
# import os
# import argparse
# from pathlib import Path

# # bpy is only available inside Blender — guarded below
# try:
#     import bpy
#     IN_BLENDER = True
# except ImportError:
#     IN_BLENDER = False


# def parse_args():
#     """
#     Parse arguments passed after the '--' separator in the Blender command.
#     sys.argv looks like: ['blender', ..., '--', '--seq', 'Seq1', ...]
#     """
#     argv = sys.argv
#     if "--" in argv:
#         argv = argv[argv.index("--") + 1:]
#     else:
#         argv = []

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seq",    required=True, help="Sequence name, e.g. Seq1")
#     parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
#     parser.add_argument("--debug",  action="store_true", help="Render debug overlays")
#     return parser.parse_args(argv)


# def setup_scene(cfg: dict):
#     """
#     One-time Blender scene setup:
#       - Clear the default scene (cube, camera, light)
#       - Set render resolution and engine from config
#       - Add a world/environment (plain dark background for Tesla look)
#     """
#     # TODO: implement
#     # import bpy
#     # bpy.ops.object.select_all(action='SELECT')
#     # bpy.ops.object.delete()
#     #
#     # scene = bpy.context.scene
#     # scene.render.engine         = cfg["blender"]["render_engine"]
#     # scene.render.resolution_x   = cfg["blender"]["resolution"][0]
#     # scene.render.resolution_y   = cfg["blender"]["resolution"][1]
#     # scene.render.film_transparent = True
#     # if cfg["blender"]["render_engine"] == "CYCLES":
#     #     scene.cycles.samples = cfg["blender"]["samples"]
#     raise NotImplementedError("setup_scene not yet implemented")


# def render_sequence(seq_name: str, cfg: dict, debug: bool = False):
#     """
#     Main loop: iterate over all detection JSONs for a sequence,
#     update the scene per frame, and render each frame to PNG.
#     """
#     return


# def main():
#     args = parse_args()

#     render_sequence(...)


# if __name__ == "__main__":
#     if not IN_BLENDER:
#         print("WARNING: bpy not found. This script must be run inside Blender.")
#         print("  blender --background --python blender/scene.py -- --seq Seq1")
#     main()
