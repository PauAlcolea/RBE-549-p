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
import re
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
        self._scene_name = None
        self._camera_name = None
        self._frame_objects = []     # track objects placed this frame for cleanup
        self._templates: Dict[str, object] = {}  # cache of linked template objects
        self._template_groups: Dict[str, list] = {}  # cache of linked grouped templates
        self._load_templates()
        self.Materials = MaterialLibrary(cfg)

    def set_render_context(self, scene_name: str = None, camera_name: str = None):
        """Store active scene/camera so style overrides can be scene-specific."""
        self._scene_name = str(scene_name) if scene_name is not None else None
        self._camera_name = str(camera_name) if camera_name is not None else None

    def _load_templates(self):
        asset_files = {
            "sedanandhatchbacks":   self.assets_dir / "Vehicles/SedanAndHatchback.blend",
            "bicycle":              self.assets_dir / "Vehicles/Bicycle.blend",
            "motorcycle":           self.assets_dir / "Vehicles/Motorcycle.blend",
            "truck":                self.assets_dir / "Vehicles/Truck.blend",
            "pickuptruck":          self.assets_dir / "Vehicles/PickupTruck.blend",
            "suv":                  self.assets_dir / "Vehicles/SUV.blend",
            "pedestrian":           self.assets_dir / "Pedestrain.blend",
            "stop_sign":            self.assets_dir / "StopSign.blend",
            "traffic_light":        self.assets_dir / "TrafficSignal.blend",
            "traffic_cone":         self.assets_dir / "TrafficConeAndCylinder.blend",
            "trash_can":            self.assets_dir / "Dustbin.blend",
            "traffic_pole":         self.assets_dir / "TrafficAssets.blend",
            "speed_limit_sign": self.assets_dir / "SpeedLimitSign.blend",
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
                preferred_parts = ("bin_mesh", "lid_mesh", "wheels_mesh")
                by_name = {norm(m.name): m for m in meshes}
                selected = [by_name[p] for p in preferred_parts if p in by_name]

                # If exact names are not present, include all mesh objects.
                keep_meshes = selected if selected else list(meshes)

                # Stable order for consistent transforms/material behavior.
                keep_meshes = sorted(keep_meshes, key=lambda m: norm(m.name))
                self._template_groups[name] = keep_meshes
                mesh_obj = keep_meshes[0]

                if len(meshes) > 1:
                    mesh_names = [m.name for m in meshes]
                    chosen_names = [m.name for m in keep_meshes]
                    print(f"[assets] trash_can mesh group: all={mesh_names} chosen={chosen_names}")

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
        Instantiate the detected vehicle asset at the vehicle's 3D position.

        Parameters
        ----------
        vehicle : dict with keys "position_3d", "depth_m"
        """
        vehicle_class = str(vehicle.get("class", "car")).lower()
        asset_name = self._vehicle_asset_name(vehicle_class)
        obj = self._instance(asset_name)
        bpos = self._json_to_blender(vehicle["position_3d"])
        obj.location = bpos
        vehicle_scale = self._vehicle_scale(vehicle_class)
        obj.scale = vehicle_scale

        heading_rad = vehicle.get("heading_rad")
        if heading_rad is not None:
            obj.rotation_euler[2] = (
                -(float(heading_rad) - math.pi / 2)
                + self._vehicle_yaw_offset(vehicle_class)
            )
            if vehicle_class == "suv":
                obj.rotation_euler[2] += math.pi
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        self._frame_objects.append(obj)
        print(
            f"[assets] vehicle: class={vehicle_class} asset={asset_name} scale={tuple(vehicle_scale)} "
            f"json_pos={vehicle['position_3d']}  →  blender_pos={bpos}  depth={vehicle['depth_m']:.1f}m"
        )

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
        Instantiate the stop sign asset.

        Apply STOP text texture directly to the imported sign mesh.
        This avoids decal geometry/orientation artifacts while keeping text visible.
        """
        obj = self._instance("stop_sign")
        obj.location = self._json_to_blender(sign["position_3d"])
        obj.scale = (0.5, 0.5, 0.5)
        obj.rotation_euler[2] = -math.pi / 2  # rotate to face camera diagonally
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        texture_path = Path(self.cfg["paths"]["assets_dir"]) / "StopSignImage.png"
        self._apply_stop_sign_texture_head_only(obj, texture_path)

        self._frame_objects.append(obj)
        print(f"[assets] stop sign: json_pos={sign['position_3d']}  →  blender_pos={obj.location}")

    def _apply_stop_sign_texture_head_only(self, obj, texture_path: Path):
        """Apply stop-sign texture to sign head only, keeping pole untextured."""
        self._apply_sign_texture_head_only(obj, texture_path, z_cut_ratio=0.55)

    def _apply_sign_texture_head_only(
        self,
        obj,
        texture_path: Path,
        z_cut_ratio=0.55,
        uv_projector=None,
        min_head_area_ratio=None,
    ):
        """Apply a sign texture to upper faces only, preserving lower pole material."""
        base_mat = obj.data.materials[0] if len(obj.data.materials) > 0 else None

        self.Materials.apply_texture(obj, texture_path)
        tex_mat = bpy.data.materials.get(f"tex_{texture_path.stem}")
        if tex_mat is None:
            return

        tex_idx = None
        base_idx = None
        for idx, mat in enumerate(obj.data.materials):
            if mat is tex_mat:
                tex_idx = idx
            if base_mat is not None and mat is base_mat:
                base_idx = idx

        if tex_idx is None:
            obj.data.materials.append(tex_mat)
            tex_idx = len(obj.data.materials) - 1

        if base_idx is None:
            if base_mat is not None:
                obj.data.materials.append(base_mat)
                base_idx = len(obj.data.materials) - 1
            else:
                base_idx = tex_idx

        world_z = [(obj.matrix_world @ v.co).z for v in obj.data.vertices]
        min_wz = min(world_z)
        max_wz = max(world_z)
        z_cut = min_wz + float(z_cut_ratio) * (max_wz - min_wz)

        candidate_polys = []
        for poly in obj.data.polygons:
            poly_wz = (obj.matrix_world @ poly.center).z
            if poly_wz >= z_cut:
                candidate_polys.append(poly)

        if min_head_area_ratio is not None and candidate_polys:
            area_cut = max(poly.area for poly in candidate_polys) * float(min_head_area_ratio)
            head_poly_ids = {poly.index for poly in candidate_polys if poly.area >= area_cut}
            if not head_poly_ids:
                head_poly_ids = {poly.index for poly in candidate_polys}
        else:
            head_poly_ids = {poly.index for poly in candidate_polys}

        for poly in obj.data.polygons:
            if poly.index in head_poly_ids:
                poly.material_index = tex_idx
            else:
                poly.material_index = base_idx

        if uv_projector is not None and head_poly_ids:
            uv_projector(obj, head_poly_ids)

    @staticmethod
    def _project_speed_sign_head_uv(obj, head_poly_ids):
        """Project speed-sign UVs on local X/Z to keep text upright and stable."""
        mesh = obj.data
        uv_layer = mesh.uv_layers.active
        if uv_layer is None:
            uv_layer = mesh.uv_layers.new(name="UVMap")

        head_vert_ids = set()
        for poly in mesh.polygons:
            if poly.index in head_poly_ids:
                head_vert_ids.update(poly.vertices)
        if not head_vert_ids:
            return

        coords = [mesh.vertices[i].co for i in head_vert_ids]
        x_min = min(c.x for c in coords)
        x_max = max(c.x for c in coords)
        z_min = min(c.z for c in coords)
        z_max = max(c.z for c in coords)
        dx = max(x_max - x_min, 1e-6)
        dz = max(z_max - z_min, 1e-6)

        for poly in mesh.polygons:
            if poly.index not in head_poly_ids:
                continue
            for loop_idx in poly.loop_indices:
                vid = mesh.loops[loop_idx].vertex_index
                co = mesh.vertices[vid].co
                u = (co.x - x_min) / dx
                v = (co.z - z_min) / dz
                uv_layer.data[loop_idx].uv = (u, v)

    def place_traffic_light(self, light: dict):
        """
        Instantiate the traffic light asset and set its state (red/yellow/green).
        Texture path: Data/Assets/traffic_light_texture.png (given by project).

                light dict keys consumed:
                    - position_3d: [x, y, z] in camera coordinates (required)
                    - color: red|yellow|green (optional, defaults to red)
                    - traffic_light_style:
                        standard_vertical|wide_green_arrow_candidate|square_arrow_signal_candidate
                        (optional, defaults to standard_vertical)
        """
        # Choose state color from detection if available; default to red.
        tl_color = light.get("color", "red")
        if tl_color not in {"red", "yellow", "green"}:
            tl_color = "red"

        tl_style = light.get("traffic_light_style", "standard_vertical")

        # Scene prior: in scene6, all traffic lights should use the normal
        # housing with an up-arrow glyph in the bottom (green) lens slot.
        force_scene6_up_arrow = (self._scene_name == "scene6")
        blender_pos = self._json_to_blender(light["position_3d"])

        # Guardrail for scene6: traffic lights should be overhead, not near
        # ground level. Skip obvious false positives that touch the road.
        if force_scene6_up_arrow and blender_pos[2] <= 1.0:
            print(
                f"[assets] skip low traffic light in scene6: "
                f"json_pos={light['position_3d']} blender_z={blender_pos[2]:.3f}"
            )
            return

        # Special style: render standalone lane-control symbol only.
        if tl_style == "square_arrow_signal_candidate" and not force_scene6_up_arrow:
            self._add_square_signal_symbol(
                base_location=Vector(blender_pos),
                active_color=tl_color,
            )
            print(
                f"[assets] square signal: json_pos={light['position_3d']}  "
                f"state={tl_color}  style={tl_style}"
            )
            return

        obj = self._instance("traffic_light")
        obj.location = blender_pos
        obj.scale = (1.0, 1.0, 1.0)
        obj.rotation_euler[2] = -math.pi / 2  # rotate to face the camera

        # Color the traffic-light body/housing yellow (bulbs are separate).
        body_mat = self.Materials.get_traffic_light_body_material()
        if body_mat is not None:
            obj.data.materials.clear()
            obj.data.materials.append(body_mat)

        render_green_as_arrow = (
            tl_style == "wide_green_arrow_candidate" and tl_color == "green"
        )
        arrow_direction = "right"
        if force_scene6_up_arrow:
            render_green_as_arrow = True
            arrow_direction = "up"

        # Scene6 rule: red/yellow bulbs stay off; only the green arrow is lit.
        active_color_for_render = "green" if force_scene6_up_arrow else tl_color

        # Add three emissive bulbs, with optional green-arrow replacement.
        self._add_traffic_light_bulbs(
            obj,
            active_color=active_color_for_render,
            render_green_as_arrow=render_green_as_arrow,
            arrow_direction=arrow_direction,
        )

        self._frame_objects.append(obj)
        print(
            f"[assets] traffic light: json_pos={light['position_3d']}  "
            f"→  blender_pos={obj.location}  state={tl_color}  style={tl_style}"
        )

    
    def place_traffic_cone(self, cone: dict):
        """Instantiate a traffic cone asset."""
        obj = self._instance("traffic_cone")
        obj.location = self._json_to_blender(cone["position_3d"])
        obj.scale = (1.0, 1.0, 1.0)  # adjust if the cone model is not already at the right size

        cone_mat = self.Materials.get_traffic_cone_material()
        if cone_mat is not None:
            obj.data.materials.clear()
            obj.data.materials.append(cone_mat)

        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)
        self._frame_objects.append(obj)
        print(f"[assets] traffic cone: json_pos={cone['position_3d']}  →  blender_pos={obj.location}")


    def place_trash_can(self, can: dict):
        """Instantiate a trash can asset."""
        obj, children = self._instance_group("trash_can")
        obj.location = self._json_to_blender(can["position_3d"])

        part_scales = {
            "bin_mesh": 1.0,
            "lid_mesh": 10,
            "wheels_mesh": 10,
        }
        self._scale_group_children(children, part_scales)

        part_materials = {
            "bin_mesh": self.Materials.get_trash_can_bin_material(),
            "lid_mesh": self.Materials.get_trash_can_lid_material(),
            "wheels_mesh": self.Materials.get_trash_can_wheels_material(),
        }
        self._apply_group_child_materials(children, part_materials)

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

    def place_speed_limit_sign(self, sign: dict):
        obj = self._instance("speed_limit_sign")
        obj.location = self._json_to_blender(sign["position_3d"])
        obj.scale = (1.0, 1.0, 1.0)
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        speed_value = self._resolve_speed_value(sign)
        texture_path = self.Materials.get_speed_limit_texture(speed_value)
        if texture_path is not None:
            self._apply_speed_limit_texture_head_only(obj, texture_path)

        if speed_value is not None and not self.Materials.pillow_available():
            text_obj = self._add_speed_limit_value_text(obj, speed_value)
            if text_obj is not None:
                self._frame_objects.append(text_obj)

        self._frame_objects.append(obj)
        print(
            f"[assets] speed limit sign: json_pos={sign['position_3d']}  "
            f"→  blender_pos={obj.location}  speed_value={speed_value}"
        )

    def _resolve_speed_value(self, sign: dict):
        """Resolve numeric speed from structured value first, OCR text second."""
        val = sign.get("speed_value")
        coerced = self._coerce_positive_int(val)
        if coerced is not None:
            return coerced

        raw_text = sign.get("ocr_raw_text")
        if not raw_text:
            return None

        digits = re.findall(r"\d+", str(raw_text))
        if not digits:
            return None

        # Prefer the longest token (e.g., "50" over split noise like "5" and "0").
        candidates = sorted(digits, key=lambda d: (-len(d), d))
        for token in candidates:
            parsed = self._coerce_positive_int(token)
            if parsed is None:
                continue

            ocr_cfg = self.cfg.get("perception", {}).get("speed_limit_ocr", {})
            min_v = self._coerce_positive_int(ocr_cfg.get("min_speed_mph")) or 1
            max_v = self._coerce_positive_int(ocr_cfg.get("max_speed_mph")) or 999
            if min_v <= parsed <= max_v:
                return parsed

        return None

    @staticmethod
    def _coerce_positive_int(value):
        try:
            out = int(float(value))
        except Exception:
            return None
        return out if out > 0 else None

    def _apply_speed_limit_texture_head_only(self, obj, texture_path: Path):
        """Apply speed-limit texture to sign head while preserving pole material."""
        self._apply_sign_texture_head_only(
            obj,
            texture_path,
            z_cut_ratio=0.62,
            uv_projector=self._project_speed_sign_head_uv,
            min_head_area_ratio=0.60,
        )

    @staticmethod
    def _add_speed_limit_value_text(sign_obj, speed_value):
        """Fallback: render speed value as Blender text on sign face."""
        try:
            value = int(float(speed_value))
            if value <= 0:
                return None
        except Exception:
            return None

        bpy.context.view_layer.update()
        bbox = [sign_obj.matrix_world @ Vector(corner) for corner in sign_obj.bound_box]
        min_z = min(p.z for p in bbox)
        max_z = max(p.z for p in bbox)

        # The object bbox includes the pole, so estimate the sign head from upper vertices.
        mesh_world_verts = [sign_obj.matrix_world @ v.co for v in sign_obj.data.vertices]
        if mesh_world_verts:
            z_cut = min_z + 0.62 * (max_z - min_z)
            head_pts = [p for p in mesh_world_verts if p.z >= z_cut]
        else:
            head_pts = []

        if len(head_pts) < 8:
            head_pts = bbox

        head_min_z = min(p.z for p in head_pts)
        head_max_z = max(p.z for p in head_pts)
        cx = sum(p.x for p in head_pts) / len(head_pts)
        cy = sum(p.y for p in head_pts) / len(head_pts)
        # Keep numbers in the lower panel of the sign head.
        cz = head_min_z + 0.30 * (head_max_z - head_min_z)

        normal = sign_obj.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
        if normal.length <= 1e-6:
            normal = Vector((0.0, 1.0, 0.0))
        else:
            normal.normalize()

        # Keep the text on the camera-facing side of the sign.
        cam = bpy.context.scene.camera
        center = Vector((cx, cy, cz))
        if cam is not None:
            to_cam = cam.location - center
            if to_cam.length > 1e-6 and normal.dot(to_cam.normalized()) < 0.0:
                normal = -normal

        text_curve = bpy.data.curves.new(name="SpeedLimitTextCurve", type="FONT")
        text_curve.body = str(value)
        text_curve.align_x = "CENTER"
        text_curve.align_y = "CENTER"
        text_curve.extrude = 0.003

        text_obj = bpy.data.objects.new(name="SpeedLimitText", object_data=text_curve)
        bpy.context.collection.objects.link(text_obj)
        text_obj.location = center + normal * 0.01
        text_obj.rotation_euler = normal.to_track_quat("Z", "Y").to_euler()

        height = max(head_max_z - head_min_z, 0.1)
        s = max(0.06, 0.14 * height)
        text_obj.scale = (s, s, s)

        mat = bpy.data.materials.get("speed_limit_text_black")
        if mat is None:
            mat = bpy.data.materials.new(name="speed_limit_text_black")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
                rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Roughness"), None)
                if rough_inp is not None:
                    rough_inp.default_value = 0.9

        text_obj.data.materials.clear()
        text_obj.data.materials.append(mat)
        return text_obj


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

    @staticmethod
    def _scale_group_children(objects, part_scales: dict):
        """Apply per-part scale multipliers to grouped meshes by name token."""
        for obj in objects:
            obj_name = obj.name.lower()
            factor = 1.0
            for token, multiplier in part_scales.items():
                if token in obj_name:
                    factor = float(multiplier)
                    break
            obj.scale = (obj.scale.x * factor, obj.scale.y * factor, obj.scale.z * factor)

    @staticmethod
    def _apply_group_child_materials(objects, part_materials: dict):
        """Assign materials to grouped meshes by child-name token."""
        for obj in objects:
            obj_name = obj.name.lower()
            for token, mat in part_materials.items():
                if token in obj_name and mat is not None:
                    obj.data.materials.clear()
                    obj.data.materials.append(mat)
                    break

    def _add_traffic_light_bulbs(
        self,
        obj,
        active_color: str = "red",
        render_green_as_arrow: bool = False,
        arrow_direction: str = "right",
    ):
        """Create three emissive sphere bulbs near the traffic light.

        Spheres are used instead of flat disks so visibility is robust from
        different camera angles and independent of face orientation.

        When render_green_as_arrow is True, the green slot is rendered as a
        left arrow mesh for wide-format green-arrow traffic lights.
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

        # Original bulb tuning was done for traffic-light scale 0.5.
        # Scale offsets and primitive size with the actual light scale so
        # increasing light size keeps bulbs/arrows in the correct place.
        base_light_scale = 0.5
        obj_scale = max(abs(obj.scale.x), abs(obj.scale.y), abs(obj.scale.z))
        scale_factor = obj_scale / base_light_scale if base_light_scale > 1e-6 else 1.0

        body_top_offset = 0.8 * scale_factor
        toward_cam_offset = 0.35 * scale_factor
        vertical_spacing = 0.5 * scale_factor
        radius = 0.19 * scale_factor

        bulb_specs = [
            ("red", vertical_spacing),
            ("yellow", 0.0),
            ("green", -vertical_spacing),
        ]

        for color, z_off in bulb_specs:
            pos = base + Vector((0.0, 0.0, body_top_offset)) + to_cam * toward_cam_offset + Vector((0.0, 0.0, z_off))
            if color == "green" and render_green_as_arrow:
                # Pull arrow farther toward camera so it does not get hidden
                # by the traffic-light housing at larger signal scales.
                arrow_pos = pos + to_cam * (0.18 * scale_factor)
                bulb_obj = self._create_left_arrow_mesh_object(
                    name=f"TL_Arrow_{len(self._frame_objects)}",
                    location=arrow_pos,
                    width=radius * 2.6,
                    height=radius * 1.8,
                    depth=radius * 0.65,
                )
                # Billboard arrow to camera so the shape is readable and not
                # seen edge-on at long distance.
                if cam is not None:
                    cam_dir = cam.location - arrow_pos
                    if cam_dir.length > 1e-6:
                        bulb_obj.rotation_euler = cam_dir.normalized().to_track_quat("Y", "Z").to_euler()
                        # Arrow mesh points along +X (right) in local symbol space.
                        # Rotate around local Y to remap +X -> +Z for an up arrow.
                        if arrow_direction == "up":
                            bulb_obj.rotation_euler.rotate_axis("Y", math.radians(-90.0))
                else:
                    bulb_obj.rotation_euler = obj.rotation_euler.copy()
                    if arrow_direction == "up":
                        bulb_obj.rotation_euler.rotate_axis("Y", math.radians(-90.0))
                print(f"[assets] traffic light arrow enabled: pos={arrow_pos}")
            else:
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
    def _create_left_arrow_mesh_object(
        name: str,
        location: Vector,
        width: float,
        height: float,
        depth: float,
    ):
        """Create a small right-pointing arrow mesh without bpy.ops.

        The arrow is centered at location and extruded in Y for thickness.
        """
        hw = width * 0.5
        hh = height * 0.5
        neck_x = hw * 0.15
        head_x = hw
        tail_x = -hw
        body_h = hh * 0.30
        hy = max(depth * 0.5, 1e-4)

        front = [
            Vector((head_x, 0.0, 0.0)),
            Vector((neck_x, 0.0, hh)),
            Vector((neck_x, 0.0, body_h)),
            Vector((tail_x, 0.0, body_h)),
            Vector((tail_x, 0.0, -body_h)),
            Vector((neck_x, 0.0, -body_h)),
            Vector((neck_x, 0.0, -hh)),
        ]
        back = [Vector((v.x, -hy * 2.0, v.z)) for v in front]

        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        bm = bmesh.new()

        verts_f = [bm.verts.new((v.x, hy, v.z)) for v in front]
        verts_b = [bm.verts.new((v.x, -hy, v.z)) for v in back]
        bm.verts.ensure_lookup_table()

        bm.faces.new(verts_f)
        bm.faces.new(list(reversed(verts_b)))

        n = len(verts_f)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new([
                verts_f[i],
                verts_f[j],
                verts_b[j],
                verts_b[i],
            ])

        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = location
        bpy.context.collection.objects.link(obj)
        return obj

    def _add_square_signal_symbol(self, base_location: Vector, active_color: str):
        """Render standalone symbol for square_arrow_signal_candidate.

        green  -> large down arrow
        other  -> large red X
        """
        bpy.context.view_layer.update()

        cam = bpy.context.scene.camera
        to_cam = Vector((0.0, -1.0, 0.0))
        if cam is not None:
            v = cam.location - base_location
            if v.length > 1e-6:
                to_cam = v.normalized()

        center = base_location + Vector((0.0, 0.0, 1.8)) + to_cam * 1.0
        face_rot = None
        if cam is not None:
            cam_dir = cam.location - center
            if cam_dir.length > 1e-6:
                face_rot = cam_dir.normalized().to_track_quat("Y", "Z").to_euler()

        # Real lane-control signals are illuminated symbols on a black square face.
        # Add a backplate slightly behind the symbol to avoid z-fighting.
        backplate = self._create_bar_mesh_object(
            name=f"TL_SquareBack_{len(self._frame_objects)}",
            location=center - to_cam * 0.08,
            length=3.4,
            thickness=3.4,
            depth=0.24,
        )
        if face_rot is not None:
            backplate.rotation_euler = face_rot.copy()
        backplate_mat = self.Materials.get_trash_can_lid_material()
        if backplate_mat is not None:
            backplate.data.materials.clear()
            backplate.data.materials.append(backplate_mat)
        self._frame_objects.append(backplate)

        if active_color == "green":
            symbol = self._create_down_arrow_mesh_object(
                name=f"TL_DownArrow_{len(self._frame_objects)}",
                location=center,
                width=2.2,
                height=3.0,
                depth=0.35,
            )
            mat = self.Materials.get_traffic_light_material("green")
            if mat is not None:
                symbol.data.materials.clear()
                symbol.data.materials.append(mat)

            if face_rot is not None:
                symbol.rotation_euler = face_rot.copy()

            self._frame_objects.append(symbol)
            return

        # Non-green states: render red X.
        bar_a = self._create_bar_mesh_object(
            name=f"TL_XA_{len(self._frame_objects)}",
            location=center,
            length=2.8,
            thickness=0.42,
            depth=0.30,
        )
        bar_b = self._create_bar_mesh_object(
            name=f"TL_XB_{len(self._frame_objects)}",
            location=center,
            length=2.8,
            thickness=0.42,
            depth=0.30,
        )

        if face_rot is not None:
            bar_a.rotation_euler = face_rot.copy()
            bar_b.rotation_euler = face_rot.copy()
        bar_a.rotation_euler.rotate_axis("Y", math.radians(45.0))
        bar_b.rotation_euler.rotate_axis("Y", math.radians(-45.0))

        mat = self.Materials.get_traffic_light_material("red")
        for bar in (bar_a, bar_b):
            if mat is not None:
                bar.data.materials.clear()
                bar.data.materials.append(mat)
            self._frame_objects.append(bar)

    @staticmethod
    def _create_down_arrow_mesh_object(
        name: str,
        location: Vector,
        width: float,
        height: float,
        depth: float,
    ):
        """Create a down-pointing extruded arrow mesh."""
        hw = width * 0.5
        hh = height * 0.5
        neck_z = -hh * 0.2
        shaft_hw = hw * 0.34
        hy = max(depth * 0.5, 1e-4)

        front = [
            Vector((0.0, 0.0, -hh)),
            Vector((hw, 0.0, neck_z)),
            Vector((shaft_hw, 0.0, neck_z)),
            Vector((shaft_hw, 0.0, hh)),
            Vector((-shaft_hw, 0.0, hh)),
            Vector((-shaft_hw, 0.0, neck_z)),
            Vector((-hw, 0.0, neck_z)),
        ]

        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        bm = bmesh.new()

        verts_f = [bm.verts.new((v.x, hy, v.z)) for v in front]
        verts_b = [bm.verts.new((v.x, -hy, v.z)) for v in front]
        bm.verts.ensure_lookup_table()

        bm.faces.new(verts_f)
        bm.faces.new(list(reversed(verts_b)))

        n = len(verts_f)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new([verts_f[i], verts_f[j], verts_b[j], verts_b[i]])

        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = location
        bpy.context.collection.objects.link(obj)
        return obj

    @staticmethod
    def _create_bar_mesh_object(
        name: str,
        location: Vector,
        length: float,
        thickness: float,
        depth: float,
    ):
        """Create an extruded rectangular bar in X/Z, used for red X symbol."""
        hx = length * 0.5
        hz = thickness * 0.5
        hy = max(depth * 0.5, 1e-4)

        front = [
            Vector((-hx, 0.0, -hz)),
            Vector((hx, 0.0, -hz)),
            Vector((hx, 0.0, hz)),
            Vector((-hx, 0.0, hz)),
        ]

        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        bm = bmesh.new()

        verts_f = [bm.verts.new((v.x, hy, v.z)) for v in front]
        verts_b = [bm.verts.new((v.x, -hy, v.z)) for v in front]
        bm.verts.ensure_lookup_table()

        bm.faces.new(verts_f)
        bm.faces.new(list(reversed(verts_b)))
        bm.faces.new([verts_f[0], verts_f[1], verts_b[1], verts_b[0]])
        bm.faces.new([verts_f[1], verts_f[2], verts_b[2], verts_b[1]])
        bm.faces.new([verts_f[2], verts_f[3], verts_b[3], verts_b[2]])
        bm.faces.new([verts_f[3], verts_f[0], verts_b[0], verts_b[3]])

        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = location
        bpy.context.collection.objects.link(obj)
        return obj

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
    def _vehicle_asset_name(vehicle_class: str) -> str:
        """
        Map detected vehicle classes to available Blender asset keys.

        For now, all car-like classes share the sedan/hatchback asset.
        """
        asset_map = {
            "car": "sedanandhatchbacks",
            "sedan": "sedanandhatchbacks",
            "sedanandhatchbacks": "sedanandhatchbacks",
            "hatchback": "sedanandhatchbacks",
            "suv": "suv",
            "pickuptruck": "pickuptruck",
            "pickup_truck": "pickuptruck",
            "bicycle": "bicycle",
            "motorcycle": "motorcycle",
            "truck": "truck",
            "bus": "truck",
        }
        return asset_map.get(vehicle_class, "sedanandhatchbacks")

    @staticmethod
    def _vehicle_scale(vehicle_class: str) -> tuple:
        """Map detected vehicle classes to per-asset Blender scales."""
        scale_map = {
            "car": (0.02, 0.02, 0.02),
            "sedan": (0.02, 0.02, 0.02),
            "sedanandhatchbacks": (0.02, 0.02, 0.02),
            "hatchback": (0.02, 0.02, 0.02),
            "suv": (3.354, 3.354, 3.354),
            "pickuptruck": (0.5, 0.5, 0.5),
            "pickup_truck": (0.5, 0.5, 0.5),
            "truck": (0.001, 0.001, 0.001),
            "bus": (0.001, 0.001, 0.001),
            "bicycle": (0.118, 0.118, 0.118),
            "motorcycle": (0.006, 0.006, 0.006),
        }
        return scale_map.get(vehicle_class, (0.02, 0.02, 0.02))

    @staticmethod
    def _vehicle_yaw_offset(vehicle_class: str) -> float:
        """
        Per-asset yaw correction for meshes whose local forward axis differs
        from the rest of the vehicle library.
        """
        yaw_offset_map = {
            "truck": math.pi,
            "bus": math.pi,
        }
        return yaw_offset_map.get(vehicle_class, 0.0)

    @staticmethod
    def _json_to_blender(pos: list) -> tuple:
        """
        Convert JSON camera-space [X, Y, Z] to Blender world-space (x, y, z).
        JSON: X=right, Y=down, Z=forward
        Blender: X=right, Y=forward, Z=up
        """
        x, y, z = pos
        return (x, z, -y)
