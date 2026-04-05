"""
blender/assets.py
=================
Loads the provided .blend asset files and places them in the 3D scene
at positions derived from the per-frame detection JSON.

Asset inventory (from Data/Assets/):
  Phase 1: generic car, pedestrian (rigged), stop sign (with texture)
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
from mathutils import Vector, Matrix
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
        pedestrian_cfg = cfg["blender"].get("pedestrians", {})
        self.pedestrian_render_mode = str(pedestrian_cfg.get("render_mode", "skeleton")).lower()
        self.pose_score_threshold = float(pedestrian_cfg.get("keypoint_score_threshold", 0.2))
        self.pose_joint_radius_m = float(pedestrian_cfg.get("joint_radius_m", 0.045))
        self.pose_bone_radius_m = float(pedestrian_cfg.get("bone_radius_m", 0.022))
        self._frame_objects = []     # track objects placed this frame for cleanup
        self._templates: Dict[str, object] = {}  # cache of linked template objects
        self._template_groups: Dict[str, list] = {}  # cache of linked grouped templates
        self._load_templates()
        self.Materials = MaterialLibrary(cfg)

    # ── Template loading ──────────────────────────────────────────────────

    def _load_templates(self):
        asset_files = {
            "sedanandhatchbacks":   self.assets_dir / "Vehicles/SedanAndHatchback.blend",
            "bicycle":              self.assets_dir / "Vehicles/Bicycle.blend",
            "motorcycle":           self.assets_dir / "Vehicles/Motorcycle.blend",
            "truck":                self.assets_dir / "Vehicles/Truck.blend",
            "pickuptruck":          self.assets_dir / "Vehicles/PickupTruck.blend",
            "suv":                  self.assets_dir / "Vehicles/SUV.blend",
            "stop_sign":            self.assets_dir / "StopSign.blend",
            "traffic_light":        self.assets_dir / "TrafficSignal.blend",
            "traffic_cone":         self.assets_dir / "TrafficConeAndCylinder.blend",
            "trash_can":            self.assets_dir / "Dustbin.blend",
            "traffic_pole":         self.assets_dir / "TrafficAssets.blend",
        }

        for name, path in asset_files.items():
            with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
                data_to.objects = list(data_from.objects)

            meshes = [obj for obj in data_to.objects if obj is not None and obj.type == "MESH"]
            if not meshes:
                print(f"[assets] WARNING: no MESH object found in {path}, objects: {[o.name for o in data_to.objects if o]}")
                continue

            mesh_obj = meshes[0]
            keep_meshes = [mesh_obj]

            def norm(obj_name: str) -> str:
                return obj_name.lower().replace(" ", "_")

            if name == "trash_can":
                preferred_parts = ("bin_mesh", "lid_mesh", "wheels_mesh")
                by_name = {norm(m.name): m for m in meshes}
                selected = [by_name[p] for p in preferred_parts if p in by_name]
                keep_meshes = selected if selected else list(meshes)
                keep_meshes = sorted(keep_meshes, key=lambda m: norm(m.name))
                self._template_groups[name] = keep_meshes
                mesh_obj = keep_meshes[0]
                if len(meshes) > 1:
                    print(f"[assets] trash_can mesh group: all={[m.name for m in meshes]} chosen={[m.name for m in keep_meshes]}")

            if name == "traffic_pole":
                iron_poles = [m for m in meshes if norm(m.name) == "iron_pole" or "iron_pole" in norm(m.name)]
                if iron_poles:
                    mesh_obj = iron_poles[0]
                    keep_meshes = [mesh_obj]
                if len(meshes) > 1:
                    print(f"[assets] traffic_pole mesh selection: chose '{mesh_obj.name}' from {[m.name for m in meshes]}")

            for obj in list(data_to.objects):
                if obj is not None and obj not in keep_meshes:
                    bpy.data.objects.remove(obj, do_unlink=True)

            for obj in keep_meshes:
                obj.hide_render = True
                obj.hide_viewport = True

            self._templates[name] = mesh_obj

        # ── Pedestrian: load mesh + armature together ──────────────────────
        self._load_pedestrian_templates()

    def _load_pedestrian_templates(self):
        """
        Load the rigged pedestrian blend file, keeping the mesh and its armature.
        The mesh is stored under 'pedestrian', the armature under 'pedestrian_armature'.
        """
        ped_path = self.assets_dir / "RiggedPedestrian.blend"
        if not ped_path.exists():
            print(f"[assets] WARNING: pedestrian blend not found at {ped_path}")
            return

        with bpy.data.libraries.load(str(ped_path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)

        ped_mesh = None
        ped_armature = None

        for obj in data_to.objects:
            if obj is None:
                continue
            if obj.type == "MESH" and ped_mesh is None:
                ped_mesh = obj
            elif obj.type == "ARMATURE" and ped_armature is None:
                ped_armature = obj
            else:
                # Remove Camera, Light, and any extra objects we don't need
                bpy.data.objects.remove(obj, do_unlink=True)

        if ped_mesh is None:
            print("[assets] WARNING: no MESH found in RiggedPedestrian.blend")
            return

        ped_mesh.hide_render = True
        ped_mesh.hide_viewport = True
        self._templates["pedestrian"] = ped_mesh
        print(f"[assets] pedestrian mesh loaded: '{ped_mesh.name}'  scale={tuple(round(v,4) for v in ped_mesh.scale)}")

        if ped_armature is None:
            print("[assets] WARNING: no ARMATURE found in RiggedPedestrian.blend — pose driving disabled")
            return

        ped_armature.hide_render = True
        ped_armature.hide_viewport = True
        self._templates["pedestrian_armature"] = ped_armature
        bones = [b.name for b in ped_armature.data.bones]
        print(f"[assets] pedestrian armature loaded: '{ped_armature.name}'  bones={bones}")

    # ── Public placement methods ──────────────────────────────────────────

    def place_vehicle(self, vehicle: dict):
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
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        self._frame_objects.append(obj)
        print(
            f"[assets] vehicle: class={vehicle_class} asset={asset_name} scale={tuple(vehicle_scale)} "
            f"json_pos={vehicle['position_3d']}  →  blender_pos={bpos}  depth={vehicle['depth_m']:.1f}m"
        )

    def place_pedestrian(self, ped: dict):
        """Instantiate the pedestrian asset and/or render its lifted skeleton."""
        mode = self.pedestrian_render_mode
        has_skeleton = bool(ped.get("keypoints_3d_camera"))

        if mode in {"mesh", "hybrid"} or (mode == "skeleton" and not has_skeleton):
            self._place_pedestrian_mesh(ped)

        if mode in {"skeleton", "hybrid"} and has_skeleton:
            self._place_pedestrian_skeleton(ped)

    def _place_pedestrian_mesh(self, ped: dict):
        """
        Instantiate the rigged pedestrian mesh and drive its armature pose
        from the COCO-17 3D keypoints when available.
        """
        mesh_obj, arm_obj = self._instance_pedestrian_pair()

        has_kps = bool(ped.get("keypoints_3d_camera")) and arm_obj is not None
        if has_kps:
            self._drive_coco_pose(arm_obj, ped)
        else:
            # Fallback: static figure at detected position
            pos = self._json_to_blender(ped["position_3d"])
            if arm_obj is not None:
                arm_obj.location = Vector(pos)
            else:
                mesh_obj.location = Vector(pos)
            if arm_obj is None:
                self._align_object_to_ground(mesh_obj, ground_z=0.0, clearance=self.ground_clearance_m)

        self._frame_objects.append(mesh_obj)
        if arm_obj is not None:
            self._frame_objects.append(arm_obj)

        print(f"[assets] pedestrian mesh placed  pose_driven={has_kps}")

    def _instance_pedestrian_pair(self):
        """
        Duplicate the pedestrian mesh AND its armature, then re-link the
        Armature modifier on the mesh copy so it points to the new armature
        instance (not the hidden template).

        Returns (mesh_obj, arm_obj).  arm_obj is None if no armature template.
        """
        mesh_template = self._templates.get("pedestrian")
        arm_template = self._templates.get("pedestrian_armature")

        if mesh_template is None:
            raise RuntimeError("[assets] pedestrian template not loaded")

        # Duplicate mesh
        mesh_copy = mesh_template.copy()
        mesh_copy.data = mesh_template.data.copy()
        mesh_copy.hide_render = False
        mesh_copy.hide_viewport = False
        bpy.context.collection.objects.link(mesh_copy)

        if arm_template is None:
            return mesh_copy, None

        # Duplicate armature
        arm_copy = arm_template.copy()
        arm_copy.data = arm_template.data.copy()
        arm_copy.hide_render = False
        arm_copy.hide_viewport = False
        bpy.context.collection.objects.link(arm_copy)

        # Re-point the Armature modifier on the mesh copy to the new armature
        for mod in mesh_copy.modifiers:
            if mod.type == "ARMATURE":
                mod.object = arm_copy
                break

        return mesh_copy, arm_copy

    def _drive_coco_pose(self, arm_obj, ped_dict: dict):
        """
        Drive the armature pose from COCO-17 3D keypoints.

        Bone mapping (Mixamo rig ← COCO-17 indices):
          Hips        ← midpoint of kp11 + kp12, oriented hip→shoulder
          Spine       ← same direction as hip→shoulder
          Chest       ← same direction
          LeftArm     ← kp5 → kp7  (left shoulder → left elbow)
          LeftForeArm ← kp7 → kp9  (left elbow → left wrist)
          RightArm    ← kp6 → kp8
          RightForeArm← kp8 → kp10
          LeftUpLeg   ← kp11 → kp13 (left hip → left knee)
          LeftLeg     ← kp13 → kp15 (left knee → left ankle)
          RightUpLeg  ← kp12 → kp14
          RightLeg    ← kp14 → kp16
        Face bones (Neck, Head) are left in rest pose intentionally.

        Ground alignment: the armature object is translated in Z so the
        lowest ankle keypoint lands at ground_clearance_m above Z=0.
        """

        kps_raw = ped_dict.get("keypoints_3d_camera", [])
        scores  = ped_dict.get("keypoint_scores", [])

        # ── Convert valid keypoints to Blender world-space Vectors ──────────
        kps = []
        for i, kp in enumerate(kps_raw):
            if kp is None:
                kps.append(None)
                continue
            score = float(scores[i]) if i < len(scores) else 1.0
            if score < self.pose_score_threshold:
                kps.append(None)
            else:
                kps.append(Vector(self._json_to_blender(kp)))
        while len(kps) < 17:
            kps.append(None)

        # ── Derived anchor points ────────────────────────────────────────────
        def mid(a, b):
            va, vb = kps[a], kps[b]
            if va is not None and vb is not None:
                return (va + vb) * 0.5
            return va if va is not None else vb  # whichever is available, or None

        def direction(a, b):
            """World-space direction vector from kps[a] to kps[b], or None."""
            if kps[a] is not None and kps[b] is not None:
                d = kps[b] - kps[a]
                return d if d.length > 1e-4 else None
            return None

        hip_mid      = mid(11, 12) or Vector(self._json_to_blender(ped_dict["position_3d"]))
        shoulder_mid = mid(5, 6)
        spine_dir    = (shoulder_mid - hip_mid).normalized() if shoulder_mid is not None else Vector((0, 0, 1))

        # ── Ground-alignment lift ────────────────────────────────────────────
        # Compute how far to lift the armature object so ankles sit on the ground.
        ankle_zs = [kps[idx].z for idx in (15, 16) if kps[idx] is not None]
        if ankle_zs:
            lift = self.ground_clearance_m - min(ankle_zs)
        else:
            # Fallback: estimate ground from hip height (assume hip ≈ half body height)
            lift = self.ground_clearance_m - (hip_mid.z - 0.9)

        # ── Helper: build 4×4 world-space bone matrix ────────────────────────
        def bone_matrix(head: Vector, dir_vec: Vector) -> Matrix:
            """
            4×4 matrix with translation=head, +Y column pointing along dir_vec.
            roll_ref chooses the X/Z axes to minimise unwanted roll.
            """
            y = dir_vec.normalized()
            roll_ref = Vector((0, 0, 1))
            if abs(y.dot(roll_ref)) > 0.98:          # near-vertical bone
                roll_ref = Vector((1, 0, 0))
            x = roll_ref.cross(y).normalized()
            z = x.cross(y).normalized()
            return Matrix((
                (x.x, y.x, z.x, head.x),
                (x.y, y.y, z.y, head.y),
                (x.z, y.z, z.z, head.z),
                (0.0, 0.0, 0.0, 1.0),
            ))

        # ── Helper: set a pose bone's world matrix ───────────────────────────
        def aim(bone_name: str, dir_vec: Vector):
            """
            Rotate bone_name so its +Y axis points along dir_vec.
            The bone's head position is preserved (read from pb.head after
            the parent's view_layer.update()).
            """
            if dir_vec is None:
                return
            pb = arm_obj.pose.bones.get(bone_name)
            if pb is None:
                print(f"[assets] WARNING: bone '{bone_name}' not found in armature")
                return
            pb.rotation_mode = "QUATERNION"
            head = Vector(pb.head)   # world-space, valid after view_layer.update()
            pb.matrix = bone_matrix(head, dir_vec)

        def aim_root(bone_name: str, world_pos: Vector, dir_vec: Vector):
            """Set root bone: translate to world_pos AND orient along dir_vec."""
            pb = arm_obj.pose.bones.get(bone_name)
            if pb is None:
                print(f"[assets] WARNING: root bone '{bone_name}' not found")
                return
            pb.rotation_mode = "QUATERNION"
            pb.matrix = bone_matrix(world_pos, dir_vec)

        # ── Initial armature object state ────────────────────────────────────
        # Keep the object at world origin; all world positioning goes through
        # the Hips bone matrix.  The lift is applied to arm_obj.location.z
        # AFTER all bone matrices are set (it shifts the whole rig uniformly).
        arm_obj.location     = Vector((0.0, 0.0, 0.0))
        arm_obj.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()

        # ── Drive bones in strict parent-first order ─────────────────────────

        # ROOT — position hip centre, orient along spine
        aim_root("Hips", hip_mid, spine_dir)
        bpy.context.view_layer.update()

        # SPINE CHAIN (children of Hips → Spine → Chest)
        aim("Spine", spine_dir)
        bpy.context.view_layer.update()
        aim("Chest", spine_dir)
        bpy.context.view_layer.update()

        # LEFT ARM (children of Chest)
        aim("LeftArm",     direction(5, 7))   # shoulder → elbow
        bpy.context.view_layer.update()
        aim("LeftForeArm", direction(7, 9))   # elbow → wrist
        bpy.context.view_layer.update()

        # RIGHT ARM (children of Chest)
        aim("RightArm",     direction(6, 8))
        bpy.context.view_layer.update()
        aim("RightForeArm", direction(8, 10))
        bpy.context.view_layer.update()

        # LEFT LEG (children of Hips)
        aim("LeftUpLeg", direction(11, 13))   # hip → knee
        bpy.context.view_layer.update()
        aim("LeftLeg",   direction(13, 15))   # knee → ankle
        bpy.context.view_layer.update()

        # RIGHT LEG (children of Hips)
        aim("RightUpLeg", direction(12, 14))
        bpy.context.view_layer.update()
        aim("RightLeg",   direction(14, 16))
        bpy.context.view_layer.update()

        # ── Apply ground-alignment lift to the whole rig ─────────────────────
        # Moving arm_obj.location shifts all bone world-positions uniformly
        # without affecting any of the matrix_basis values we just set.
        arm_obj.location.z += lift
        bpy.context.view_layer.update()

        print(
            f"[assets] coco pose driven: arm={arm_obj.name}  "
            f"hip_world={tuple(round(v,2) for v in hip_mid)}  lift={lift:.3f}m"
        )

    # ── Skeleton overlay (unchanged) ─────────────────────────────────────

    def _place_pedestrian_skeleton(self, ped: dict):
        points_3d = ped.get("keypoints_3d_camera") or []
        scores = ped.get("keypoint_scores") or []
        skeleton_links = ped.get("skeleton_links") or []
        if not points_3d:
            return

        valid_points = []
        for idx, pt in enumerate(points_3d):
            if pt is None or len(pt) < 3:
                valid_points.append(None)
                continue
            score = float(scores[idx]) if idx < len(scores) else 1.0
            if score < self.pose_score_threshold:
                valid_points.append(None)
                continue
            valid_points.append(Vector(self._json_to_blender(pt)))

        z_values = [point.z for point in valid_points if point is not None]
        if z_values:
            z_offset = (0.0 + self.ground_clearance_m) - min(z_values)
            valid_points = [
                (point + Vector((0.0, 0.0, z_offset))) if point is not None else None
                for point in valid_points
            ]

        joint_mat = self.Materials.get_pose_joint_material()
        bone_mat  = self.Materials.get_pose_bone_material()

        for idx, point in enumerate(valid_points):
            if point is None:
                continue
            joint = self._create_sphere_mesh_object(
                name=f"ped_pose_joint_{idx}",
                location=point,
                radius=self.pose_joint_radius_m,
            )
            if joint_mat is not None:
                joint.data.materials.clear()
                joint.data.materials.append(joint_mat)
            self._frame_objects.append(joint)

        for edge in skeleton_links:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            a, b = int(edge[0]), int(edge[1])
            if a >= len(valid_points) or b >= len(valid_points):
                continue
            pa, pb = valid_points[a], valid_points[b]
            if pa is None or pb is None:
                continue
            bone = self._create_cylinder_between_points(
                name=f"ped_pose_bone_{a}_{b}",
                point_a=pa,
                point_b=pb,
                radius=self.pose_bone_radius_m,
            )
            if bone_mat is not None:
                bone.data.materials.clear()
                bone.data.materials.append(bone_mat)
            self._frame_objects.append(bone)

        print(f"[assets] pedestrian skeleton: joints={sum(p is not None for p in valid_points)}")

    # ── Other asset placements (unchanged) ───────────────────────────────

    def place_stop_sign(self, sign: dict):
        obj = self._instance("stop_sign")
        obj.location = self._json_to_blender(sign["position_3d"])
        obj.scale = (0.5, 0.5, 0.5)
        obj.rotation_euler[2] = -math.pi / 2
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        texture_path = Path(self.cfg["paths"]["assets_dir"]) / "StopSignImage.png"
        decal_obj = self._create_stop_sign_decal(obj)
        self.Materials.apply_texture(decal_obj, texture_path)
        self._frame_objects.append(decal_obj)
        self._frame_objects.append(obj)
        print(f"[assets] stop sign: json_pos={sign['position_3d']}  →  blender_pos={obj.location}")

    def place_traffic_light(self, light: dict):
        obj = self._instance("traffic_light")
        obj.location = self._json_to_blender(light["position_3d"])
        obj.scale = (0.5, 0.5, 0.5)
        obj.rotation_euler[2] = -math.pi / 2

        body_mat = self.Materials.get_traffic_light_body_material()
        if body_mat is not None:
            obj.data.materials.clear()
            obj.data.materials.append(body_mat)

        tl_color = light.get("color", "red")
        if tl_color not in {"red", "yellow", "green"}:
            tl_color = "red"

        self._add_traffic_light_bulbs(obj, active_color=tl_color)
        self._frame_objects.append(obj)
        print(f"[assets] traffic light: json_pos={light['position_3d']}  →  blender_pos={obj.location}  state={tl_color}")

    def place_traffic_cone(self, cone: dict):
        obj = self._instance("traffic_cone")
        obj.location = self._json_to_blender(cone["position_3d"])
        obj.scale = (1.0, 1.0, 1.0)

        cone_mat = self.Materials.get_traffic_cone_material()
        if cone_mat is not None:
            obj.data.materials.clear()
            obj.data.materials.append(cone_mat)

        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)
        self._frame_objects.append(obj)
        print(f"[assets] traffic cone: json_pos={cone['position_3d']}  →  blender_pos={obj.location}")

    def place_trash_can(self, can: dict):
        obj, children = self._instance_group("trash_can")
        obj.location = self._json_to_blender(can["position_3d"])

        part_scales = {"bin_mesh": 1.0, "lid_mesh": 10, "wheels_mesh": 10}
        self._scale_group_children(children, part_scales)

        part_materials = {
            "bin_mesh":    self.Materials.get_trash_can_bin_material(),
            "lid_mesh":    self.Materials.get_trash_can_lid_material(),
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

    def clear_frame_objects(self):
        """Delete all objects placed during the previous frame."""
        for obj in self._frame_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        self._frame_objects.clear()

        for obj in bpy.data.objects:
            if obj.name.startswith("vehicle") or obj.name.startswith("pedestrian"):
                bpy.data.objects.remove(obj, do_unlink=True)

    # ── Internal helpers ──────────────────────────────────────────────────

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
        for obj in objects:
            obj_name = obj.name.lower()
            for token, mat in part_materials.items():
                if token in obj_name and mat is not None:
                    obj.data.materials.clear()
                    obj.data.materials.append(mat)
                    break

    def _add_traffic_light_bulbs(self, obj, active_color: str = "red"):
        bpy.context.view_layer.update()
        base = obj.location.copy()
        cam = bpy.context.scene.camera

        to_cam = Vector((0.0, -1.0, 0.0))
        if cam is not None:
            v = cam.location - base
            if v.length > 1e-6:
                to_cam = v.normalized()

        toward_cam_offset = 0.35
        vertical_spacing = 0.5
        radius = 0.19

        bulb_specs = [
            ("red",    vertical_spacing),
            ("yellow", 0.0),
            ("green",  -vertical_spacing),
        ]

        for color, z_off in bulb_specs:
            pos = base + Vector((0.0, 0.0, 0.8)) + to_cam * toward_cam_offset + Vector((0.0, 0.0, z_off))
            bulb_obj = self._create_sphere_mesh_object(
                name=f"TL_Bulb_{color}",
                location=pos,
                radius=radius,
            )
            mat_key = color if color == active_color else "unknown"
            mat = self.Materials.get_traffic_light_material(mat_key)
            if mat is not None:
                bulb_obj.data.materials.clear()
                bulb_obj.data.materials.append(mat)
            self._frame_objects.append(bulb_obj)

    @staticmethod
    def _create_sphere_mesh_object(name: str, location: Vector, radius: float):
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=radius)
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(name, mesh)
        obj.location = location
        bpy.context.collection.objects.link(obj)
        return obj

    @staticmethod
    def _create_cylinder_between_points(name: str, point_a: Vector, point_b: Vector, radius: float):
        direction = point_b - point_a
        length = direction.length
        if length <= 1e-6:
            return AssetLibrary._create_sphere_mesh_object(name, point_a, radius)

        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=12,
                              radius1=radius, radius2=radius, depth=length)
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(name, mesh)
        obj.location = (point_a + point_b) / 2.0
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction.normalized())
        bpy.context.collection.objects.link(obj)
        return obj

    def _create_stop_sign_decal(self, stop_obj):
        bpy.context.view_layer.update()
        world_corners = [stop_obj.matrix_world @ Vector(corner) for corner in stop_obj.bound_box]
        min_x = min(c.x for c in world_corners)
        max_x = max(c.x for c in world_corners)
        max_z = max(c.z for c in world_corners)

        width = max(max_x - min_x, 1e-4)
        radius = max(0.18, 0.52 * width)
        center = Vector(((min_x + max_x) * 0.5, stop_obj.location.y, max_z - 1.05 * radius))

        cam = bpy.context.scene.camera
        if cam is not None:
            to_cam = cam.location - center
            if to_cam.length > 1e-6:
                center += to_cam.normalized() * 0.02

        mesh = bpy.data.meshes.new("StopSignDecal_Mesh")
        verts = [
            (-radius, 0.0, -radius),
            ( radius, 0.0, -radius),
            ( radius, 0.0,  radius),
            (-radius, 0.0,  radius),
        ]
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        mesh.update()

        uv_layer = mesh.uv_layers.new(name="UVMap")
        for loop_idx, uv in zip(mesh.polygons[0].loop_indices,
                                [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]):
            uv_layer.data[loop_idx].uv = uv

        decal_obj = bpy.data.objects.new("StopSignDecal", mesh)
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
        bpy.context.view_layer.update()
        min_z = min((obj.matrix_world @ Vector(corner)).z for corner in obj.bound_box)
        obj.location.z += (ground_z + clearance) - min_z

    @staticmethod
    def _align_group_to_ground(parent_obj, objects, ground_z: float = 0.0, clearance: float = 0.0):
        bpy.context.view_layer.update()
        min_z = min(
            (obj.matrix_world @ Vector(corner)).z
            for obj in objects
            for corner in obj.bound_box
        )
        parent_obj.location.z += (ground_z + clearance) - min_z

    # ── Static lookup tables (unchanged) ─────────────────────────────────

    @staticmethod
    def _vehicle_yaw_from_direction(direction: str) -> float:
        yaw_map = {
            "right": -math.pi / 2,
            "left":   math.pi / 2,
            "approaching_right": -math.pi / 4,
            "approaching_left":   math.pi / 4,
            "receding_right":    -3 * math.pi / 4,
            "receding_left":      3 * math.pi / 4,
            "approaching": 0.0,
            "receding":    math.pi,
            "stationary":  0.0,
            "unknown":     0.0,
        }
        return yaw_map.get(direction, 0.0)

    @staticmethod
    def _vehicle_asset_name(vehicle_class: str) -> str:
        asset_map = {
            "car":               "sedanandhatchbacks",
            "sedan":             "sedanandhatchbacks",
            "sedanandhatchbacks":"sedanandhatchbacks",
            "hatchback":         "sedanandhatchbacks",
            "suv":               "suv",
            "pickuptruck":       "pickuptruck",
            "pickup_truck":      "pickuptruck",
            "bicycle":           "bicycle",
            "motorcycle":        "motorcycle",
            "truck":             "truck",
            "bus":               "truck",
        }
        return asset_map.get(vehicle_class, "sedanandhatchbacks")

    @staticmethod
    def _vehicle_scale(vehicle_class: str) -> tuple:
        scale_map = {
            "car":               (0.02, 0.02, 0.02),
            "sedan":             (0.02, 0.02, 0.02),
            "sedanandhatchbacks":(0.02, 0.02, 0.02),
            "hatchback":         (0.02, 0.02, 0.02),
            "suv":               (3.354, 3.354, 3.354),
            "pickuptruck":       (0.5, 0.5, 0.5),
            "pickup_truck":      (0.5, 0.5, 0.5),
            "truck":             (0.001, 0.001, 0.001),
            "bus":               (0.001, 0.001, 0.001),
            "bicycle":           (0.118, 0.118, 0.118),
            "motorcycle":        (0.006, 0.006, 0.006),
        }
        return scale_map.get(vehicle_class, (0.02, 0.02, 0.02))

    @staticmethod
    def _vehicle_yaw_offset(vehicle_class: str) -> float:
        yaw_offset_map = {
            "truck": math.pi,
            "bus":   math.pi,
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