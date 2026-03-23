# """
# blender/assets.py
# =================
# Loads the provided .blend asset files and places them in the 3D scene
# at positions derived from the per-frame detection JSON.

# Asset inventory (from Data/Assets/):
#   Phase 1: generic car, pedestrian, stop sign (with texture)
#   Phase 2: sedan, SUV, truck, bicycle, motorcycle, traffic cone, pole, dustbin

# Coordinate system note:
#   The JSON gives 3D positions in camera space (X right, Y down, Z forward).
#   Blender uses right-handed Z-up. camera.py sets up the camera so that:
#     Blender X =  JSON X
#     Blender Y =  JSON Z  (forward = Blender Y)
#     Blender Z = -JSON Y  (down flipped to up)
# """

# from pathlib import Path
# from typing import Dict


# class AssetLibrary:
#     """
#     Manages loading and instancing of Blender assets.

#     Usage
#     -----
#     lib = AssetLibrary(cfg)
#     lib.place_vehicle(vehicle_dict)
#     lib.place_pedestrian(ped_dict)
#     lib.place_stop_sign(sign_dict)
#     lib.clear_frame_objects()    # call before each new frame
#     """

#     def __init__(self, cfg: dict):
#         self.cfg = cfg
#         self.assets_dir = Path(cfg["paths"]["assets_dir"])
#         self._frame_objects = []     # track objects placed this frame for cleanup
#         self._templates: Dict[str, object] = {}  # cache of linked template objects
#         self._load_templates()

#     def _load_templates(self):
#         """
#         Link all asset .blend files into the scene as hidden template objects.
#         We'll instance (duplicate) them for each detection.
#         """
#         # TODO: implement
#         # import bpy
#         # asset_files = {
#         #     "car":       self.assets_dir / "Sedan.blend",
#         #     "pedestrian":self.assets_dir / "Pedestrian.blend",
#         #     "stop_sign": self.assets_dir / "StopSign.blend",
#         # }
#         # for name, path in asset_files.items():
#         #     with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
#         #         data_to.objects = [data_from.objects[0]]
#         #     obj = data_to.objects[0]
#         #     obj.hide_render = True
#         #     obj.hide_viewport = True
#         #     self._templates[name] = obj
#         pass

#     def place_vehicle(self, vehicle: dict):
#         """
#         Instantiate the car asset at the vehicle's 3D position.

#         Parameters
#         ----------
#         vehicle : dict with keys "position_3d", "depth_m"
#         """
#         # TODO: implement
#         # obj = self._instance("car")
#         # obj.location = self._json_to_blender(vehicle["position_3d"])
#         # self._frame_objects.append(obj)
#         pass

#     def place_pedestrian(self, ped: dict):
#         """Instantiate the pedestrian asset."""
#         # TODO: implement
#         # obj = self._instance("pedestrian")
#         # obj.location = self._json_to_blender(ped["position_3d"])
#         # self._frame_objects.append(obj)
#         pass

#     def place_stop_sign(self, sign: dict):
#         """
#         Instantiate the stop sign asset and apply the provided texture.
#         Texture path: Data/Assets/stop_sign_texture.png (given by project).
#         """
#         # TODO: implement
#         # obj = self._instance("stop_sign")
#         # obj.location = self._json_to_blender(sign["position_3d"])
#         # materials.apply_texture(obj, self.assets_dir / "stop_sign_texture.png")
#         # self._frame_objects.append(obj)
#         pass

#     def clear_frame_objects(self):
#         """Delete all objects placed during the previous frame."""
#         # TODO: implement
#         # import bpy
#         # for obj in self._frame_objects:
#         #     bpy.data.objects.remove(obj, do_unlink=True)
#         self._frame_objects.clear()

#     # ── Helpers ───────────────────────────────────────────────────────────

#     def _instance(self, name: str):
#         """Duplicate a template object and link it to the scene."""
#         # TODO: implement
#         # import bpy
#         # template = self._templates[name]
#         # new_obj = template.copy()
#         # new_obj.data = template.data.copy()
#         # new_obj.hide_render = False
#         # new_obj.hide_viewport = False
#         # bpy.context.collection.objects.link(new_obj)
#         # return new_obj
#         raise NotImplementedError

#     @staticmethod
#     def _json_to_blender(pos: list) -> tuple:
#         """
#         Convert JSON camera-space [X, Y, Z] to Blender world-space (x, y, z).
#         JSON: X=right, Y=down, Z=forward
#         Blender: X=right, Y=forward, Z=up
#         """
#         x, y, z = pos
#         return (x, z, -y)
