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

import math
from pathlib import Path
from typing import Dict
import bpy
import bmesh
from mathutils import Vector
from .materials import MaterialLibrary


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
        self.ground_clearance_m = cfg["blender"].get("ground_clearance_m", 0.03)
        self._frame_objects = []     # track objects placed this frame for cleanup
        self._templates: Dict[str, object] = {}  # cache of linked template objects
        self._template_groups: Dict[str, list] = {}  # cache of linked grouped templates
        self._load_templates()
        self.Materials = MaterialLibrary(cfg)

    def _load_templates(self):
        asset_files = {
            "car":           self.assets_dir / "Vehicles/SedanAndHatchback.blend",
            "pedestrian":    self.assets_dir / "Pedestrain.blend",
            "stop_sign":     self.assets_dir / "StopSign.blend",
            "traffic_light": self.assets_dir / "TrafficSignal.blend",
            "traffic_cone":  self.assets_dir / "TrafficConeAndCylinder.blend",
            "trash_can":     self.assets_dir / "Dustbin.blend",
            "traffic_pole":  self.assets_dir / "TrafficAssets.blend", # iron pole
        }

        for name, path in asset_files.items():
            with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
                # Load ALL objects from the file so we can pick the right one
                data_to.objects = list(data_from.objects)

            meshes = [obj for obj in data_to.objects if obj is not None and obj.type == "MESH"]
            if not meshes:
                print(f"[assets] WARNING: no MESH object found in {path}, objects: {[o.name for o in data_to.objects if o]}")
                continue

            mesh_obj = meshes[0]
            keep_meshes = [mesh_obj]

            def norm(obj_name: str) -> str:
                return obj_name.lower().replace(" ", "_")

            # Dustbin.blend contains multiple meshes (bin/lid/wheels).
            # Keep the full set and instance as one grouped object.
            if name == "trash_can":
                keep_meshes = list(meshes)
                self._template_groups[name] = keep_meshes
                mesh_obj = keep_meshes[0]

                if len(meshes) > 1:
                    mesh_names = [m.name for m in meshes]
                    print(f"[assets] trash_can mesh group: {mesh_names}")

            if name == "traffic_pole":
                iron_poles = [m for m in meshes if norm(m.name) == "iron_pole" or "iron_pole" in norm(m.name)]
                if iron_poles:
                    mesh_obj = iron_poles[0]
                    keep_meshes = [mesh_obj]
                if len(meshes) > 1:
                    mesh_names = [m.name for m in meshes]
                    print(f"[assets] traffic_pole mesh selection: chose '{mesh_obj.name}' from {mesh_names}")

            # Hide all loaded objects, keep only selected mesh(es).
            for obj in list(data_to.objects):
                if obj is not None and obj not in keep_meshes:
                    bpy.data.objects.remove(obj, do_unlink=True)

            for obj in keep_meshes:
                obj.hide_render = True
                obj.hide_viewport = True

            self._templates[name] = mesh_obj

    def place_vehicle(self, vehicle: dict):
        """
        Instantiate the car asset at the vehicle's 3D position.

        Parameters
        ----------
        vehicle : dict with keys "position_3d", "depth_m"
        """
        obj = self._instance("car")
        bpos = self._json_to_blender(vehicle["position_3d"])
        obj.location = bpos
        obj.scale = (0.02, 0.02, 0.02)

        direction = vehicle.get("direction", "unknown")
        obj.rotation_euler[2] = self._vehicle_yaw_from_direction(direction)
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        self._frame_objects.append(obj)
        print(f"[assets] vehicle: json_pos={vehicle['position_3d']}  →  blender_pos={bpos}  depth={vehicle['depth_m']:.1f}m")

    def place_pedestrian(self, ped: dict):
        """Instantiate the pedestrian asset."""
        obj = self._instance("pedestrian")
        obj.location = self._json_to_blender(ped["position_3d"])
        obj.scale = (0.009, 0.009, 0.009)
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)
        self._frame_objects.append(obj)
        print(f"[assets] pedestrian: json_pos={ped['position_3d']}  →  blender_pos={obj.location}")

    def place_stop_sign(self, sign: dict):
        """
        Instantiate the stop sign asset and apply the provided texture.
        Texture path: Data/Assets/stop_sign_texture.png (given by project).
        """
        obj = self._instance("stop_sign")
        obj.location = self._json_to_blender(sign["position_3d"])
        obj.scale = (0.5, 0.5, 0.5)
        obj.rotation_euler[2] = -math.pi / 2  # rotate to face camera diagonally
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        texture_path = Path(self.cfg["paths"]["assets_dir"]) / "StopSignImage.png"
        decal_obj = self._create_stop_sign_decal(obj)
        self.Materials.apply_texture(decal_obj, texture_path)
        self._frame_objects.append(decal_obj)

        self._frame_objects.append(obj)
        print(f"[assets] stop sign: json_pos={sign['position_3d']}  →  blender_pos={obj.location}")

    def place_traffic_light(self, light: dict):
        """
        Instantiate the traffic light asset and set its state (red/yellow/green).
        Texture path: Data/Assets/traffic_light_texture.png (given by project).
        """
        obj = self._instance("traffic_light")
        obj.location = self._json_to_blender(light["position_3d"])
        obj.scale = (0.5, 0.5, 0.5)
        obj.rotation_euler[2] = -math.pi / 2  # rotate to face the camera

        # Choose state color from detection if available; default to red.
        tl_color = light.get("color", "red")
        if tl_color not in {"red", "yellow", "green"}:
            tl_color = "red"

        # Add three emissive "bulb" disks near the traffic light position.
        self._add_traffic_light_bulbs(obj, active_color=tl_color)

        self._frame_objects.append(obj)
        print(f"[assets] traffic light: json_pos={light['position_3d']}  →  blender_pos={obj.location}  state={tl_color}")

    
    def place_traffic_cone(self, cone: dict):
        """Instantiate a traffic cone asset."""
        obj = self._instance("traffic_cone")
        obj.location = self._json_to_blender(cone["position_3d"])
        obj.scale = (1.0, 1.0, 1.0)  # adjust if the cone model is not already at the right size
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)
        self._frame_objects.append(obj)
        print(f"[assets] traffic cone: json_pos={cone['position_3d']}  →  blender_pos={obj.location}")


    def place_trash_can(self, can: dict):
        """Instantiate a trash can asset."""
        obj, children = self._instance_group("trash_can")
        obj.location = self._json_to_blender(can["position_3d"])
        obj.scale = (0.2, 0.2, 0.2)
        obj.rotation_euler[2] = -math.pi / 2
        self._align_group_to_ground(obj, children, ground_z=0.0, clearance=self.ground_clearance_m)
        self._frame_objects.append(obj)
        self._frame_objects.extend(children)
        print(f"[assets] trash can: json_pos={can['position_3d']}  →  blender_pos={obj.location}")

    def place_traffic_pole(self, pole: dict):
        obj = self._instance("traffic_pole")
        obj.location = self._json_to_blender(pole["position_3d"])
        obj.scale = (0.1, 0.1, 0.4)
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)
        self._frame_objects.append(obj)
        print(f"[assets] traffic pole: json_pos={pole['position_3d']}  →  blender_pos={obj.location}")


    def clear_frame_objects(self):
        """Delete all objects placed during the previous frame."""
        for obj in self._frame_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        self._frame_objects.clear()

        for obj in bpy.data.objects:
            if obj.name.startswith("vehicle") or obj.name.startswith("pedestrian"):
                bpy.data.objects.remove(obj, do_unlink=True)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _instance(self, name: str):
        """Duplicate a template object and link it to the scene."""
        template = self._templates[name]
        new_obj = template.copy()
        new_obj.data = template.data.copy()
        new_obj.hide_render = False
        new_obj.hide_viewport = False
        bpy.context.collection.objects.link(new_obj)
        return new_obj

    def _instance_group(self, name: str):
        """Duplicate a grouped template and return (root, children)."""
        templates = self._template_groups[name]
        root = bpy.data.objects.new(f"{name}_group", None)
        root.empty_display_type = "PLAIN_AXES"
        bpy.context.collection.objects.link(root)

        children = []
        for template in templates:
            child = template.copy()
            child.data = template.data.copy()
            child.hide_render = False
            child.hide_viewport = False
            bpy.context.collection.objects.link(child)
            child.parent = root
            child.matrix_parent_inverse = root.matrix_world.inverted()
            children.append(child)

        return root, children

    def _add_traffic_light_bulbs(self, obj, active_color: str = "red"):
        """Create three emissive sphere bulbs near the traffic light.

        Spheres are used instead of flat disks so visibility is robust from
        different camera angles and independent of face orientation.
        """

        bpy.context.view_layer.update()

        base = obj.location.copy()
        cam = bpy.context.scene.camera

        # Move bulbs slightly toward camera so they are not hidden by the
        # traffic light mesh.
        to_cam = Vector((0.0, -1.0, 0.0))
        if cam is not None:
            v = cam.location - base
            if v.length > 1e-6:
                to_cam = v.normalized()

        toward_cam_offset = 0.35
        vertical_spacing = 0.5
        radius = 0.19

        bulb_specs = [
            ("red", vertical_spacing),
            ("yellow", 0.0),
            ("green", -vertical_spacing),
        ]

        for color, z_off in bulb_specs:
            pos = base + Vector((0.0, 0.0, 0.8)) + to_cam * toward_cam_offset + Vector((0.0, 0.0, z_off))
            bulb_obj = self._create_sphere_mesh_object(
                name=f"TL_Bulb_{color}",
                location=pos,
                radius=radius,
            )

            # Use the traffic-light color only for the active bulb;
            # others use the "unknown" grey for an "off" look.
            mat_key = color if color == active_color else "unknown"
            mat = self.Materials.get_traffic_light_material(mat_key)
            if mat is not None:
                bulb_obj.data.materials.clear()
                bulb_obj.data.materials.append(mat)

            self._frame_objects.append(bulb_obj)

    @staticmethod
    def _create_sphere_mesh_object(name: str, location: Vector, radius: float):
        """Create a small UV-sphere mesh object without bpy.ops."""
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(
            bm,
            u_segments=16,
            v_segments=8,
            radius=radius,
        )
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = location
        bpy.context.collection.objects.link(obj)
        return obj

    def _create_stop_sign_decal(self, stop_obj):
        """Create a textured octagon decal in front of the stop-sign head.

        The decal has clean UVs and is independent of the imported asset's
        material slots/UV layout, which avoids clipped edges and texture bleed.
        """
        bpy.context.view_layer.update()

        world_corners = [stop_obj.matrix_world @ Vector(corner) for corner in stop_obj.bound_box]
        min_x = min(c.x for c in world_corners)
        max_x = max(c.x for c in world_corners)
        max_z = max(c.z for c in world_corners)

        width = max(max_x - min_x, 1e-4)
        # Slightly oversize so the decal fully covers the white sign face.
        radius = max(0.18, 0.52 * width)

        center = Vector(((min_x + max_x) * 0.5, stop_obj.location.y, max_z - 1.05 * radius))

        cam = bpy.context.scene.camera
        if cam is not None:
            to_cam = cam.location - center
            if to_cam.length > 1e-6:
                center += to_cam.normalized() * 0.02

        mesh = bpy.data.meshes.new("StopSignDecal_Mesh")

        # Build a quad in local XZ plane (normal +Y). The PNG alpha defines
        # the octagon silhouette, while the quad guarantees robust UV mapping.
        verts = [
            (-radius, 0.0, -radius),
            ( radius, 0.0, -radius),
            ( radius, 0.0,  radius),
            (-radius, 0.0,  radius),
        ]
        faces = [(0, 1, 2, 3)]
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        # Full-range UV mapping with U flipped so STOP reads correctly.
        uv_layer = mesh.uv_layers.new(name="UVMap")
        uv_coords = [
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ]
        for loop_idx, uv in zip(mesh.polygons[0].loop_indices, uv_coords):
            uv_layer.data[loop_idx].uv = uv

        decal_obj = bpy.data.objects.new("StopSignDecal", mesh)

        # Billboard the decal toward camera so orientation does not depend on
        # the imported asset's local axis conventions.
        cam = bpy.context.scene.camera
        if cam is not None:
            to_cam = cam.location - center
            if to_cam.length > 1e-6:
                dir_cam = to_cam.normalized()
                decal_obj.rotation_euler = dir_cam.to_track_quat("Y", "Z").to_euler()
                decal_obj.location = center + dir_cam * 0.12
            else:
                decal_obj.rotation_euler = stop_obj.rotation_euler.copy()
                decal_obj.location = center
        else:
            decal_obj.rotation_euler = stop_obj.rotation_euler.copy()
            decal_obj.location = center

        bpy.context.collection.objects.link(decal_obj)
        return decal_obj

    @staticmethod
    def _align_object_to_ground(obj, ground_z: float = 0.0, clearance: float = 0.0):
        """
        Lift an object so its lowest world-space bbox point sits on the ground.

        This makes placement much more robust when the asset origin is not at
        the wheel/foot contact plane.
        """
        from mathutils import Vector

        bpy.context.view_layer.update()
        min_z = min((obj.matrix_world @ Vector(corner)).z for corner in obj.bound_box)
        obj.location.z += (ground_z + clearance) - min_z

    @staticmethod
    def _align_group_to_ground(parent_obj, objects, ground_z: float = 0.0, clearance: float = 0.0):
        """Lift a grouped object so the group's lowest world-space point sits on ground."""
        from mathutils import Vector

        bpy.context.view_layer.update()
        min_z = min((obj.matrix_world @ Vector(corner)).z for obj in objects for corner in obj.bound_box)
        parent_obj.location.z += (ground_z + clearance) - min_z

    @staticmethod
    def _vehicle_yaw_from_direction(direction: str) -> float:
        """
        Map coarse motion labels to a plausible vehicle yaw in Blender.

        Convention:
          - 0 rad      : facing away from camera / forward in scene
          - pi rad     : facing toward camera
          - +/- pi/2   : side-facing left/right
        """
        yaw_map = {
            "right": -math.pi / 2,
            "left": math.pi / 2,
            "approaching_right": -math.pi / 4,
            "approaching_left": math.pi / 4,
            "receding_right": -3 * math.pi / 4,
            "receding_left": 3 * math.pi / 4,
            "approaching": 0.0,
            "receding": math.pi,
            "stationary": 0.0,
            "unknown": 0.0,
        }
        return yaw_map.get(direction, 0.0)

    @staticmethod
    def _json_to_blender(pos: list) -> tuple:
        """
        Convert JSON camera-space [X, Y, Z] to Blender world-space (x, y, z).
        JSON: X=right, Y=down, Z=forward
        Blender: X=right, Y=forward, Z=up
        """
        x, y, z = pos
        return (x, z, -y)
