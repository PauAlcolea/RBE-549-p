"""
blender/assets.py
=================
Loads the provided .blend asset files and places them in the 3D scene
at positions derived from the per-frame detection JSON.

Asset inventory (from Data/Assets/):
  Phase 1: generic car, pedestrian, stop sign (with texture)
  Phase 2: sedan, SUV, truck, bicycle, motorcycle, traffic cone, pole, dustbin

Coordinate system note:
  The JSON gives 3D positions in camera space (X right, Y down, Z forward).
  Blender uses right-handed Z-up. camera.py sets up the camera so that:
    Blender X =  JSON X
    Blender Y =  JSON Z  (forward = Blender Y)
    Blender Z = -JSON Y  (down flipped to up)
"""

from pathlib import Path
from typing import Dict


class AssetLibrary:
    """
    Manages loading and instancing of Blender assets.

    Usage
    -----
    lib = AssetLibrary(cfg)
    lib.place_vehicle(vehicle_dict)
    lib.place_pedestrian(ped_dict)
    lib.place_stop_sign(sign_dict)
    lib.clear_frame_objects()    # call before each new frame
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.assets_dir = Path(cfg["paths"]["assets_dir"])
        self._frame_objects = []     # track objects placed this frame for cleanup
        self._templates: Dict[str, object] = {}  # cache of linked template objects
        self._load_templates()

    def _load_templates(self):
        import bpy
        asset_files = {
            "car":           self.assets_dir / "Vehicles/SedanAndHatchback.blend",
            "pedestrian":    self.assets_dir / "Pedestrain.blend",
            "stop_sign":     self.assets_dir / "StopSign.blend",
            "traffic_light": self.assets_dir / "TrafficSignal.blend",
        }

        for name, path in asset_files.items():
            with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
                # Load ALL objects from the file so we can pick the right one
                data_to.objects = list(data_from.objects)

            # Find the first MESH object (skip lights, cameras, empties)
            mesh_obj = None
            for obj in data_to.objects:
                if obj is not None and obj.type == "MESH":
                    mesh_obj = obj
                    break

            if mesh_obj is None:
                print(f"[assets] WARNING: no MESH object found in {path}, objects: {[o.name for o in data_to.objects if o]}")
                continue

            # Hide all loaded objects, keep only the mesh template
            for obj in data_to.objects:
                if obj is not None and obj is not mesh_obj:
                    bpy.data.objects.remove(obj, do_unlink=True)

            mesh_obj.hide_render = True
            mesh_obj.hide_viewport = True
            self._templates[name] = mesh_obj

    def place_vehicle(self, vehicle: dict):
        """
        Instantiate the car asset at the vehicle's 3D position.

        Parameters
        ----------
        vehicle : dict with keys "position_3d", "depth_m"
        """
        obj = self._instance("car")
        obj.location = self._json_to_blender(vehicle["position_3d"])
        self._frame_objects.append(obj)

    def place_pedestrian(self, ped: dict):
        """Instantiate the pedestrian asset."""
        obj = self._instance("pedestrian")
        obj.location = self._json_to_blender(ped["position_3d"])
        self._frame_objects.append(obj)
        pass

    def place_stop_sign(self, sign: dict):
        """
        Instantiate the stop sign asset and apply the provided texture.
        Texture path: Data/Assets/stop_sign_texture.png (given by project).
        """
        # TODO: implement
        # obj = self._instance("stop_sign")
        # obj.location = self._json_to_blender(sign["position_3d"])
        # materials.apply_texture(obj, self.assets_dir / "StopSignImage.png")
        # self._frame_objects.append(obj)
        pass

    def clear_frame_objects(self):
        """Delete all objects placed during the previous frame."""
        import bpy
        for obj in self._frame_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        self._frame_objects.clear()

        # for obj in bpy.data.objects:
        #     if obj.name.startswith("vehicle") or obj.name.startswith("pedestrian"):
        #         bpy.data.objects.remove(obj, do_unlink=True)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _instance(self, name: str):
        """Duplicate a template object and link it to the scene."""
        import bpy
        template = self._templates[name]
        new_obj = template.copy()
        new_obj.data = template.data.copy()
        new_obj.hide_render = False
        new_obj.hide_viewport = False
        bpy.context.collection.objects.link(new_obj)
        return new_obj

    @staticmethod
    def _json_to_blender(pos: list) -> tuple:
        """
        Convert JSON camera-space [X, Y, Z] to Blender world-space (x, y, z).
        JSON: X=right, Y=down, Z=forward
        Blender: X=right, Y=forward, Z=up
        """
        x, y, z = pos
        return (x, z, -y)
