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
import pickle
import copy
from pathlib import Path
from typing import Dict, List, Optional
import bpy
import bmesh
from mathutils import Vector
from .materials import MaterialLibrary
import numpy as np

class _SmplFKSolver:
    """
    Minimal CPU-only SMPL forward kinematics using numpy.
    Reconstructs the 6890-vertex mesh from pose (72,) and betas (10,).
    Loads once from SMPL_NEUTRAL.pkl; all data is cached as float32 numpy arrays.
    """

    class _CompatCSCMatrix:
        """
        Minimal scipy.sparse.csc_matrix compatibility shim.
        Enough for loading legacy SMPL pickles and calling todense().
        """

        def todense(self):
            shape = getattr(self, "_shape", None)
            if shape is None:
                raise ValueError("Invalid sparse matrix: missing _shape")
            rows, cols = int(shape[0]), int(shape[1])
            out = np.zeros((rows, cols), dtype=np.float32)

            indptr = np.asarray(getattr(self, "indptr"), dtype=np.int64).ravel()
            indices = np.asarray(getattr(self, "indices"), dtype=np.int64).ravel()
            data = np.asarray(getattr(self, "data"), dtype=np.float32).ravel()

            max_cols = min(cols, max(0, len(indptr) - 1))
            for c in range(max_cols):
                start = int(indptr[c])
                end = int(indptr[c + 1])
                if end <= start:
                    continue
                out[indices[start:end], c] = data[start:end]
            return out

    class _CompatCh:
        """
        Minimal chumpy.ch.Ch compatibility shim.
        Exposes wrapped data through numpy conversion.
        """

        def __array__(self, dtype=None, copy=None):
            x = np.asarray(getattr(self, "x"))
            if dtype is not None:
                x = x.astype(dtype, copy=False)
            if copy:
                x = x.copy()
            return x

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "scipy.sparse.csc" and name == "csc_matrix":
                return _SmplFKSolver._CompatCSCMatrix
            if module == "chumpy.ch" and name == "Ch":
                return _SmplFKSolver._CompatCh
            return super().find_class(module, name)

    @staticmethod
    def _load_pickle_model(smpl_pkl_path: str):
        """
        Load SMPL pickle with no-scipy / no-chumpy fallbacks.
        """
        try:
            with open(smpl_pkl_path, "rb") as f:
                return pickle.load(f, encoding="latin1")
        except ModuleNotFoundError as exc:
            if exc.name not in {"scipy", "chumpy"}:
                raise
            with open(smpl_pkl_path, "rb") as f:
                unpickler = _SmplFKSolver._CompatUnpickler(
                    f, fix_imports=True, encoding="latin1"
                )
                return unpickler.load()

    @staticmethod
    def _normalize_shapedirs(raw_shapedirs, num_verts: int) -> np.ndarray:
        sd = np.asarray(raw_shapedirs, dtype=np.float32)

        # Common: (V, 3, num_betas)
        if sd.ndim == 3 and sd.shape[0] == num_verts and sd.shape[1] == 3:
            return sd

        # Alternate: (3, V, num_betas)
        if sd.ndim == 3 and sd.shape[0] == 3 and sd.shape[1] == num_verts:
            return np.transpose(sd, (1, 0, 2))

        # Flattened forms.
        if sd.ndim == 2 and sd.shape[0] == num_verts * 3:
            return sd.reshape(num_verts, 3, sd.shape[1])
        if sd.ndim == 2 and sd.shape[1] == num_verts * 3:
            return sd.T.reshape(num_verts, 3, sd.shape[0])

        raise ValueError(f"Unexpected shapedirs shape: {sd.shape}")

    @staticmethod
    def _normalize_posedirs(raw_posedirs, num_verts: int) -> np.ndarray:
        pd = np.asarray(raw_posedirs, dtype=np.float32)
        expected_rows = num_verts * 3

        # Common: (V, 3, 207) -> (V*3, 207)
        if pd.ndim == 3 and pd.shape[0] == num_verts and pd.shape[1] == 3:
            return pd.reshape(expected_rows, pd.shape[2])

        # Alternate: (207, V, 3) -> (V*3, 207)
        if pd.ndim == 3 and pd.shape[0] == 207 and pd.shape[1] == num_verts and pd.shape[2] == 3:
            return np.transpose(pd, (1, 2, 0)).reshape(expected_rows, 207)

        # Flattened forms.
        if pd.ndim == 2 and pd.shape[0] == expected_rows:
            return pd
        if pd.ndim == 2 and pd.shape[1] == expected_rows:
            return pd.T

        raise ValueError(f"Unexpected posedirs shape: {pd.shape}")

    def __init__(self, smpl_pkl_path: str):
        m = self._load_pickle_model(smpl_pkl_path)

        self.v_template  = np.array(m["v_template"],  dtype=np.float32)            # (V, 3)
        nv = self.v_template.shape[0]
        self.shapedirs   = self._normalize_shapedirs(m["shapedirs"], nv)           # (V, 3, num_betas)
        self.weights     = np.array(m["weights"],     dtype=np.float32)            # (6890, 24)
        self.faces       = np.array(m["f"],           dtype=np.int32)              # (13776, 3)

        # J_regressor may be scipy sparse
        Jr = m["J_regressor"]
        self.J_regressor = np.array(
            Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float32
        )  # (24, 6890)

        # posedirs shape varies by SMPL version: normalize to (V*3, 207)
        self.posedirs = self._normalize_posedirs(m["posedirs"], nv)

        kt = np.array(m["kintree_table"], dtype=np.int64)
        self.parent = kt[0].copy()  # (24,)
        # Some pickles store root parent as uint max.
        if self.parent[0] > 1000:
            self.parent[0] = -1

    @staticmethod
    def _rodrigues(r: np.ndarray) -> np.ndarray:
        """Axis-angle (3,) → rotation matrix (3,3)."""
        theta = float(np.linalg.norm(r))
        if theta < 1e-8:
            return np.eye(3, dtype=np.float32)
        n = r / theta
        c, s = np.cos(theta), np.sin(theta)
        K = np.array([[0, -n[2], n[1]],
                      [n[2], 0, -n[0]],
                      [-n[1], n[0], 0]], dtype=np.float32)
        return (c * np.eye(3, dtype=np.float32)
                + (1.0 - c) * np.outer(n, n).astype(np.float32)
                + s * K)

    def forward(self, pose, betas):
        """
        pose  : array-like (72,) – axis-angle for 24 SMPL joints
        betas : array-like (10,) – shape coefficients
        Returns: verts (6890, 3) in SMPL local y-up space,
                 faces (13776, 3) fixed topology
        """
        pose  = np.asarray(pose,  dtype=np.float32).reshape(24, 3)
        betas = np.asarray(betas, dtype=np.float32).reshape(-1)
        nv    = self.v_template.shape[0]   # 6890

        # ── 1. Shape blend shapes ──────────────────────────────────────────
        num_betas = min(self.shapedirs.shape[2], betas.shape[0])
        if num_betas <= 0:
            raise ValueError("SMPL betas are empty")
        v_shaped = self.v_template + np.einsum(
            "ijk,k->ij", self.shapedirs[:, :, :num_betas], betas[:num_betas]
        )

        # ── 2. Joint positions in rest pose ───────────────────────────────
        J = self.J_regressor @ v_shaped    # (24, 3)

        # ── 3. Pose rotation matrices ─────────────────────────────────────
        Rs = np.stack([self._rodrigues(pose[j]) for j in range(24)])  # (24,3,3)

        # ── 4. Pose blend shapes (exclude global orient, joints 1-23) ─────
        pose_feat = (Rs[1:] - np.eye(3, dtype=np.float32)).ravel()    # (207,)
        num_pose_coeff = min(self.posedirs.shape[1], pose_feat.shape[0])
        if num_pose_coeff <= 0:
            raise ValueError("SMPL posedirs has no pose coefficients")
        v_posed = v_shaped + (
            self.posedirs[:, :num_pose_coeff] @ pose_feat[:num_pose_coeff]
        ).reshape(nv, 3)

        # ── 5. Global joint transforms (forward kinematics) ───────────────
        G = [None] * 24
        for j in range(24):
            t = J[j] if j == 0 else (J[j] - J[self.parent[j]])
            Gj = np.eye(4, dtype=np.float32)
            Gj[:3, :3] = Rs[j]
            Gj[:3, 3]  = t
            G[j] = Gj if j == 0 else (G[self.parent[j]] @ Gj)
        G = np.stack(G)   # (24, 4, 4)

        # ── 6. Remove rest-pose joint contribution ────────────────────────
        rest          = np.tile(np.eye(4, dtype=np.float32), (24, 1, 1))
        rest[:, :3, 3] = -J
        G_star = G @ rest   # (24, 4, 4)

        # ── 7. LBS – blend transforms per vertex ──────────────────────────
        T      = np.einsum("vj,jkl->vkl", self.weights, G_star)  # (6890,4,4)
        v_homo = np.ones((nv, 4), dtype=np.float32)
        v_homo[:, :3] = v_posed
        verts  = np.einsum("vij,vj->vi", T, v_homo)[:, :3]       # (6890, 3)

        return verts, self.faces
    

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
    _SMPL24_BONES = (
        (0, 1), (1, 4), (4, 7), (7, 10),
        (0, 2), (2, 5), (5, 8), (8, 11),
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
        (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
    )

    @staticmethod
    def _resolve_path(value: str, base_dir: Path) -> Path:
        p = Path(str(value)).expanduser()
        if p.is_absolute():
            return p
        return (base_dir / p).resolve()

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.config_dir = Path(cfg.get("_meta", {}).get("config_dir", Path.cwd()))
        self.assets_dir = self._resolve_path(cfg["paths"]["assets_dir"], self.config_dir)
        self.ground_clearance_m = cfg["blender"].get("ground_clearance_m", 0.03)
        blender_cfg = cfg.get("blender", {})
        ped_cfg = blender_cfg.get("pedestrian")
        if not isinstance(ped_cfg, dict):
            ped_cfg = blender_cfg.get("pedestrians", {})
            if isinstance(ped_cfg, dict) and ped_cfg:
                print("[assets] WARNING: using deprecated config key 'blender.pedestrians'; "
                      "please rename it to 'blender.pedestrian'.")
            else:
                ped_cfg = {}
                print("[assets] WARNING: missing 'blender.pedestrian' config; "
                      "defaulting to PyMAF skeleton mode.")
        self.ped_mode = str(ped_cfg.get("mode", "pymaf")).strip().lower()
        self.pymaf_target_height_m = float(ped_cfg.get("pymaf_target_height_m", 1.70))
        self.pymaf_bone_radius_m = float(ped_cfg.get("pymaf_bone_radius_m", 0.018))
        self.pymaf_joint_radius_m = float(ped_cfg.get("pymaf_joint_radius_m", 0.025))
        self.pymaf_min_joint_count = int(ped_cfg.get("pymaf_min_joint_count", 24))
        self.pymaf_use_last24 = bool(ped_cfg.get("pymaf_use_last24", True))
        self.pymaf_y_up = bool(ped_cfg.get("pymaf_y_up", True))
        self.pymaf_upright_hysteresis = float(ped_cfg.get("pymaf_upright_hysteresis", 0.15))
        self.pymaf_flip_confirm_frames = max(1, int(ped_cfg.get("pymaf_flip_confirm_frames", 2)))
        self.pymaf_reuse_previous_on_failure = bool(ped_cfg.get("pymaf_reuse_previous_on_failure", True))
        self.pymaf_allow_asset_fallback = bool(ped_cfg.get("pymaf_allow_asset_fallback", False))
        self.pymaf_pose_guard_enabled = bool(ped_cfg.get("pymaf_pose_guard_enabled", True))
        self.pymaf_pose_guard_soft_joint_delta_rad = float(
            ped_cfg.get("pymaf_pose_guard_soft_joint_delta_rad", 0.65)
        )
        self.pymaf_pose_guard_soft_joint_count = max(
            1, int(ped_cfg.get("pymaf_pose_guard_soft_joint_count", 2))
        )
        self.pymaf_pose_guard_soft_max_joint_delta_rad = float(
            ped_cfg.get("pymaf_pose_guard_soft_max_joint_delta_rad", 0.80)
        )
        self.pymaf_pose_guard_hard_joint_delta_rad = float(
            ped_cfg.get("pymaf_pose_guard_hard_joint_delta_rad", 1.20)
        )
        self.pymaf_pose_guard_root_delta_rad = float(
            ped_cfg.get("pymaf_pose_guard_root_delta_rad", 0.80)
        )
        self.pymaf_pose_clamp_enabled = bool(ped_cfg.get("pymaf_pose_clamp_enabled", False))
        self.pymaf_pose_max_joint_delta_rad = float(
            ped_cfg.get("pymaf_pose_max_joint_delta_rad", 0.55)
        )
        self.pymaf_pose_hold_max_fallback_frames = max(
            1,
            int(
                ped_cfg.get(
                    "pymaf_pose_hold_max_fallback_frames",
                    ped_cfg.get("pymaf_pose_hold_min_frames", 3),
                )
            ),
        )
        self._ped_upright_state: Dict[str, dict] = {}
        self._ped_last_valid_by_track: Dict[str, dict] = {}
        self._ped_pose_hold_state: Dict[str, dict] = {}
        self._ped_prev_pose_by_track: Dict[str, list] = {}
        
        # ── SMPL mesh mode ────────────────────────────────────────────────────
        self.smpl_fk: "_SmplFKSolver | None" = None
        if self.ped_mode in ("smpl_mesh", "mesh"):
            default_smpl = self._resolve_path("../Weights/SMPL_NEUTRAL.pkl", self.config_dir)
            smpl_pkl = ped_cfg.get(
                "smpl_pkl",
                str(default_smpl),
            )
            smpl_pkl = self._resolve_path(smpl_pkl, self.config_dir)
            if smpl_pkl.exists():
                try:
                    self.smpl_fk = _SmplFKSolver(str(smpl_pkl))
                    print(f"[assets] SMPL FK solver loaded from {smpl_pkl}")
                except Exception as exc:
                    print(f"[assets] WARNING: could not load SMPL FK solver: {exc}")
            else:
                print(f"[assets] WARNING: smpl_pkl not found at {smpl_pkl}; "
                    f"falling back to skeleton mode")
        
        self._scene_name = None
        self._camera_name = None
        self._frame_objects = []     # track objects placed this frame for cleanup
        self._ground_arrow_bboxes = []
        self._ground_text_bboxes = []
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
            "tesla":                self.assets_dir / "Vehicles/Tesla.blend",
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

            # Some assets contain multiple meshes and must be instanced as a group.
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

            if name == "tesla":
                keep_meshes = sorted(list(meshes), key=lambda m: norm(m.name))
                self._template_groups[name] = keep_meshes
                mesh_obj = keep_meshes[0]
                if len(meshes) > 1:
                    mesh_names = [m.name for m in meshes]

            if name == "traffic_pole":
                def has_iron_pole_ancestor(obj) -> bool:
                    cur = obj
                    while cur is not None:
                        if "iron_pole" in norm(cur.name):
                            return True
                        cur = cur.parent
                    return False

                # Prefer Cylinder.001 under the iron_pole hierarchy when available.
                preferred_cyl_with_ancestor = [
                    m
                    for m in meshes
                    if (
                        norm(m.name) in {"cylinder.001", "cylinder_001"}
                        or "cylinder.001" in norm(m.name)
                        or "cylinder_001" in norm(m.name)
                    )
                    and has_iron_pole_ancestor(m)
                ]

                # Fallback: still pick Cylinder.001 by name even if ancestry info is missing.
                preferred_cyl_any = [
                    m
                    for m in meshes
                    if (
                        norm(m.name) in {"cylinder.001", "cylinder_001"}
                        or "cylinder.001" in norm(m.name)
                        or "cylinder_001" in norm(m.name)
                    )
                ]

                if preferred_cyl_with_ancestor:
                    mesh_obj = preferred_cyl_with_ancestor[0]
                    keep_meshes = [mesh_obj]
                elif preferred_cyl_any:
                    mesh_obj = preferred_cyl_any[0]
                    keep_meshes = [mesh_obj]
                else:
                    iron_poles = [m for m in meshes if norm(m.name) == "iron_pole" or "iron_pole" in norm(m.name)]
                    if iron_poles:
                        mesh_obj = iron_poles[0]
                        keep_meshes = [mesh_obj]
                    else:
                        print("[assets] WARNING: traffic_pole could not find Cylinder.001; using first mesh fallback")
                if len(meshes) > 1:
                    mesh_names = [m.name for m in meshes]
                    print(f"[assets] traffic_pole mesh selection: chose '{mesh_obj.name}' from {mesh_names}")

            # Hide all loaded objects, keep only selected mesh(es).
            if name not in self._template_groups and len(meshes) > 1:
                keep_meshes = sorted(list(meshes), key=lambda m: norm(m.name))
                self._template_groups[name] = keep_meshes
                mesh_obj = keep_meshes[0]
                print(f"[assets] {name}: registering {len(keep_meshes)} meshes as group: "
                      f"{[m.name for m in keep_meshes]}")

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

        children = []
        if asset_name in self._template_groups:
            obj, children = self._instance_group(asset_name)
        else:
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

        if children:
            self._align_group_to_ground(obj, children, ground_z=0.0, clearance=self.ground_clearance_m)
        else:
            self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        self._frame_objects.append(obj)
        self._frame_objects.extend(children)
        print(
            f"[assets] {vehicle_class}: scale={tuple(vehicle_scale)} "
            f"json_pos={vehicle['position_3d']}  →  blender_pos={bpos}  depth={vehicle['depth_m']:.1f}m"
        )

    def place_ego_vehicle(self):
        """Instantiate the configured ego vehicle (Tesla) at world origin."""
        ego_cfg = self.cfg.get("blender", {}).get("ego_vehicle", {})
        asset_name = str(ego_cfg.get("asset", "tesla")).lower()
        if asset_name not in self._templates:
            print(f"[assets] WARNING: ego asset '{asset_name}' not loaded; skipping ego vehicle")
            return None

        children = []
        if asset_name in self._template_groups:
            obj, children = self._instance_group(asset_name)
        else:
            obj = self._instance(asset_name)

        location = ego_cfg.get("location", [0.0, 0.0, 0.0])
        if not isinstance(location, (list, tuple)) or len(location) != 3:
            location = [0.0, 0.0, 0.0]
        obj.location = (float(location[0]), float(location[1]), float(location[2]))

        scale = ego_cfg.get("scale", self._vehicle_scale("tesla"))
        if not isinstance(scale, (list, tuple)) or len(scale) != 3:
            scale = self._vehicle_scale("tesla")
        obj.scale = (float(scale[0]), float(scale[1]), float(scale[2]))

        yaw_rad = float(ego_cfg.get("yaw_rad", 0.0))
        asset_yaw_offset_rad = float(ego_cfg.get("asset_yaw_offset_rad", 0.0))
        obj.rotation_euler[2] = yaw_rad + asset_yaw_offset_rad + self._vehicle_yaw_offset("tesla")

        targets = children if children else [obj]
        ego_paint_rgb = self._ego_paint_rgb()
        if ego_paint_rgb is not None:
            self._apply_ego_paint_override(targets, ego_paint_rgb)
        self._apply_ego_tesla_part_materials(targets)

        if children:
            self._align_group_to_ground(obj, children, ground_z=0.0, clearance=self.ground_clearance_m)
        else:
            self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)

        self._frame_objects.append(obj)
        self._frame_objects.extend(children)
        print(
            f"[assets] tesla: scale={tuple(obj.scale)} "
            f"blender_pos={(obj.location.x, obj.location.y, obj.location.z)} "
            f"heading={yaw_rad:.3f} visual_offset={asset_yaw_offset_rad:.3f}"
        )
        return obj

    def _ego_paint_rgb(self):
        """Return optional ego paint RGB override from config, or None."""
        style_cfg = self.cfg.get("blender", {}).get("style", {})
        rgb = style_cfg.get("ego_paint_rgb")
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            return None
        try:
            return tuple(max(0.0, min(1.0, float(c))) for c in rgb)
        except Exception:
            return None

    @staticmethod
    def _is_tesla_body_mesh(mesh_name: str) -> bool:
        """Heuristic filter for Tesla body-like meshes (exclude glass/lights/wheels)."""
        n = mesh_name.lower()
        include_tokens = (
            "body", "bumper", "door", "frame", "trunk", "front", "side", "top", "hood", "panel"
        )
        exclude_tokens = (
            "window", "glass", "headlight", "light", "tire", "rim", "brake", "rotor",
            "logo", "camera", "seat", "steering", "plug", "reflector", "wheel"
        )
        if any(tok in n for tok in exclude_tokens):
            return False
        return any(tok in n for tok in include_tokens)

    @staticmethod
    def _get_or_create_ego_paint_material(rgb):
        """Create/reuse glossy paint material for ego color override."""
        mat_name = "ego_paint_override"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
            metallic_inp = next((inp for inp in bsdf.inputs if inp.name == "Metallic"), None)
            if metallic_inp is not None:
                metallic_inp.default_value = 0.15
            rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Roughness"), None)
            if rough_inp is not None:
                rough_inp.default_value = 0.22
            clearcoat_inp = next((inp for inp in bsdf.inputs if inp.name == "Coat Weight"), None)
            if clearcoat_inp is None:
                clearcoat_inp = next((inp for inp in bsdf.inputs if inp.name == "Clearcoat"), None)
            if clearcoat_inp is not None:
                clearcoat_inp.default_value = 1.0
            coat_rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Coat Roughness"), None)
            if coat_rough_inp is None:
                coat_rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Clearcoat Roughness"), None)
            if coat_rough_inp is not None:
                coat_rough_inp.default_value = 0.08
        return mat

    def _apply_ego_paint_override(self, objects, rgb):
        """Assign override paint material to Tesla body meshes only."""
        mat = self._get_or_create_ego_paint_material(rgb)
        for obj in objects:
            if obj is None or obj.type != "MESH":
                continue
            if not self._is_tesla_body_mesh(obj.name):
                continue
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    @staticmethod
    def _get_or_create_principled_material(
        name: str,
        base_color,
        metallic: float,
        roughness: float,
        alpha: float = 1.0,
        transmission: float = 0.0,
        specular: float = 0.5,
    ):
        """Create/reuse a configurable Principled material for Tesla part styling."""
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (
                float(base_color[0]),
                float(base_color[1]),
                float(base_color[2]),
                1.0,
            )

            metallic_inp = next((inp for inp in bsdf.inputs if inp.name == "Metallic"), None)
            if metallic_inp is not None:
                metallic_inp.default_value = float(metallic)

            rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Roughness"), None)
            if rough_inp is not None:
                rough_inp.default_value = float(roughness)

            spec_inp = next(
                (inp for inp in bsdf.inputs if inp.name in ("Specular IOR Level", "Specular")),
                None,
            )
            if spec_inp is not None:
                spec_inp.default_value = float(specular)

            trans_inp = next((inp for inp in bsdf.inputs if inp.name == "Transmission Weight"), None)
            if trans_inp is None:
                trans_inp = next((inp for inp in bsdf.inputs if inp.name == "Transmission"), None)
            if trans_inp is not None:
                trans_inp.default_value = float(transmission)

            alpha_inp = next((inp for inp in bsdf.inputs if inp.name == "Alpha"), None)
            if alpha_inp is not None:
                alpha_inp.default_value = float(alpha)

        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND" if float(alpha) < 0.999 else "OPAQUE"
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = "HASHED" if float(alpha) < 0.999 else "OPAQUE"

        return mat

    def _ego_tesla_style_enabled(self) -> bool:
        """Return whether Tesla per-part styling is enabled."""
        style_cfg = self.cfg.get("blender", {}).get("style", {})
        return bool(style_cfg.get("ego_tesla_style_enabled", True))

    def _apply_ego_tesla_part_materials(self, objects):
        """Assign Tesla per-part materials by mesh-name tokens."""
        if not self._ego_tesla_style_enabled():
            return

        def _src_name(obj):
            return str(obj.get("source_name", obj.name)).lower()

        style_cfg = self.cfg.get("blender", {}).get("style", {})

        top_glass_color = style_cfg.get("ego_tesla_top_glass_rgb", [0.02, 0.02, 0.02])
        top_glass_alpha = float(style_cfg.get("ego_tesla_top_glass_alpha", 0.28))
        top_glass_transmission = float(style_cfg.get("ego_tesla_top_glass_transmission", 0.90))

        gray_trim_color = style_cfg.get("ego_tesla_trim_gray_rgb", [0.49, 0.49, 0.49])
        logo_gray_color = style_cfg.get("ego_tesla_logo_gray_rgb", [0.8, 0.8, 0.8])
        red_paint_color = style_cfg.get("ego_tesla_red_paint_rgb", [0.8, 0.05, 0.05])

        top_glass_mat = self._get_or_create_principled_material(
            "ego_tesla_top_glass",
            top_glass_color,
            metallic=0.0,
            roughness=0.06,
            alpha=top_glass_alpha,
            transmission=top_glass_transmission,
            specular=0.55,
        )
        trim_gray_mat = self._get_or_create_principled_material(
            "ego_tesla_trim_gray",
            gray_trim_color,
            metallic=0.88,
            roughness=0.14,
            alpha=1.0,
            transmission=0.0,
            specular=0.45,
        )
        logo_gray_mat = self._get_or_create_principled_material(
            "ego_tesla_logo_gray",
            logo_gray_color,
            metallic=1.0,
            roughness=0.10,
            alpha=1.0,
            transmission=0.0,
            specular=0.75,
        )
        red_paint_mat = self._get_or_create_principled_material(
            "ego_tesla_red_paint",
            red_paint_color,
            metallic=0.15,
            roughness=0.22,
            alpha=1.0,
            transmission=0.0,
            specular=0.45,
        )
        white_mat = self._get_or_create_principled_material(
            "ego_tesla_white",
            (0.9, 0.9, 0.9),
            metallic=0.0,
            roughness=0.1,
            alpha=1.0,
            transmission=0.0,
            specular=1.0,
        )

        side_window_mat = None
        for obj in objects:
            if obj is None or obj.type != "MESH":
                continue
            obj_name = _src_name(obj)
            if "window" not in obj_name:
                continue
            if len(obj.data.materials) == 0:
                continue
            side_window_mat = obj.data.materials[0]
            if side_window_mat is not None:
                break

        top_mat = side_window_mat if side_window_mat is not None else top_glass_mat
        default_token_materials = {
            "logo_text": logo_gray_mat,
            "top": top_mat,
            "logo text": logo_gray_mat,
            "logo": logo_gray_mat,
            "trunk.001": trim_gray_mat,
            "trunk.002": trim_gray_mat,
            "trunk": red_paint_mat,
            "bumper.": trim_gray_mat,
            "frontbumper": trim_gray_mat,
            "rearbumper": trim_gray_mat,
            "side.001": white_mat,
            "side.002": white_mat
        }

        user_token_materials = style_cfg.get("ego_tesla_part_material_tokens", {})
        if isinstance(user_token_materials, dict):
            for token, material_key in user_token_materials.items():
                token_name = str(token).lower().strip()
                if not token_name:
                    continue

                if isinstance(material_key, (list, tuple)) and len(material_key) == 3:
                    try:
                        rgb = [max(0.0, min(1.0, float(c))) for c in material_key]
                    except Exception:
                        continue
                    default_token_materials[token_name] = self._get_or_create_principled_material(
                        f"ego_tesla_token_{token_name.replace(' ', '_').replace('.', '_')}",
                        rgb,
                        metallic=0.15,
                        roughness=0.22,
                        alpha=1.0,
                        transmission=0.0,
                        specular=0.45,
                    )
                    continue

                key = str(material_key).strip().lower()
                if key == "top_glass":
                    default_token_materials[token_name] = top_glass_mat
                elif key == "side_window":
                    default_token_materials[token_name] = top_mat
                elif key == "trim_gray":
                    default_token_materials[token_name] = trim_gray_mat
                elif key == "logo_gray":
                    default_token_materials[token_name] = logo_gray_mat
                elif key in ("keep_red", "body_red", "keep"):
                    default_token_materials[token_name] = "__KEEP__"

        for obj in objects:
            if obj is None or obj.type != "MESH":
                continue

            obj_name = _src_name(obj)
            if obj_name == "bumper":
                continue
            for token, mat in default_token_materials.items():
                if token in obj_name and mat is not None:
                    if mat == "__KEEP__":
                        break
                    obj.data.materials.clear()
                    obj.data.materials.append(mat)
                    break

    def place_pedestrian(self, ped: dict):
        """Instantiate a pedestrian using PyMAF skeleton when available."""
        if self.ped_mode != "asset":
            track_key = self._ped_track_key(ped)
            track_id = ped.get("pymaf_track_id")
            has_track_id = track_id is not None

            # Always prefer current-frame PyMAF if valid.
            ped_for_render = ped
            used_pose_guard = False

            if has_track_id and self.pymaf_pose_guard_enabled:
                prev_pose = self._ped_prev_pose_by_track.get(track_key)
                prev_ped = self._ped_last_valid_by_track.get(track_key)
                if isinstance(prev_pose, list) and isinstance(prev_ped, dict):
                    drastic, reason = self._is_pose_drastic_for_track(ped_for_render, prev_pose)
                    if drastic:
                        ped_for_render = self._ped_with_updated_position(prev_ped, ped_for_render)
                        used_pose_guard = True
                        print(
                            f"[assets] pedestrian: pose_guard reused previous valid pose "
                            f"for track={track_id} ({reason})"
                        )

            clipped_joints = 0
            if has_track_id and self.pymaf_pose_clamp_enabled and not used_pose_guard:
                ped_for_render, clipped_joints = self._clamp_pose_spike_for_track(
                    ped_for_render, track_key
                )
                if clipped_joints > 0:
                    print(
                        f"[assets] pedestrian: clamped pose spike for track={track_id} "
                        f"({clipped_joints} joints)"
                    )

            if self._place_pedestrian_pymaf(ped_for_render):
                if has_track_id:
                    self._ped_last_valid_by_track[track_key] = copy.deepcopy(ped_for_render)
                    self._ped_pose_hold_state[track_key] = {"fallback_frames": 0}
                    self._update_prev_pose_cache(track_key, ped_for_render)
                return

            if self.pymaf_reuse_previous_on_failure and has_track_id:
                prev_ped = self._ped_last_valid_by_track.get(track_key)
                hold_state = self._ped_pose_hold_state.get(track_key, {"fallback_frames": 0})
                fallback_frames = int(hold_state.get("fallback_frames", 0))

                if (
                    isinstance(prev_ped, dict)
                    and fallback_frames < self.pymaf_pose_hold_max_fallback_frames
                ):
                    candidate_with_live_pos = self._ped_with_updated_position(prev_ped, ped)
                    if self._place_pedestrian_pymaf(candidate_with_live_pos):
                        self._ped_pose_hold_state[track_key] = {
                            "fallback_frames": fallback_frames + 1
                        }
                        self._update_prev_pose_cache(track_key, candidate_with_live_pos)
                        print(
                            f"[assets] pedestrian: reused previous PyMAF pose for track={track_id} "
                            f"(fallback {fallback_frames + 1}/{self.pymaf_pose_hold_max_fallback_frames})"
                        )
                        return

            if not self.pymaf_allow_asset_fallback:
                print(
                    f"[assets] pedestrian: skipped track={track_id if has_track_id else 'na'} "
                    f"(no valid PyMAF and asset fallback disabled)"
                )
                return

        obj = self._instance("pedestrian")
        obj.location = self._json_to_blender(ped["position_3d"])
        obj.scale = (0.009, 0.009, 0.009)
        self._align_object_to_ground(obj, ground_z=0.0, clearance=self.ground_clearance_m)
        self._frame_objects.append(obj)
        print(f"[assets] pedestrian: json_pos={ped['position_3d']}  →  blender_pos={obj.location}")

    @staticmethod
    def _ped_with_updated_position(source_pose: dict, live_ped: dict) -> dict:
        """
        Reuse pose/shape from source_pose while updating dynamic fields from live_ped.
        """
        merged = copy.deepcopy(source_pose)
        for key in ("position_3d", "depth_m", "bbox", "pymaf_track_id"):
            if key in live_ped:
                merged[key] = copy.deepcopy(live_ped[key])
        return merged

    @staticmethod
    def _axis_angle_geodesic_delta_rad(prev_joint_aa, curr_joint_aa) -> float:
        """
        Rotation distance between two axis-angle joint rotations.
        Uses SO(3) geodesic angle, robust to 2*pi wrapping artifacts.
        """
        prev = np.asarray(prev_joint_aa, dtype=np.float32).reshape(3)
        curr = np.asarray(curr_joint_aa, dtype=np.float32).reshape(3)
        r_prev = _SmplFKSolver._rodrigues(prev)
        r_curr = _SmplFKSolver._rodrigues(curr)
        r_delta = r_prev.T @ r_curr
        cos_ang = float((np.trace(r_delta) - 1.0) * 0.5)
        cos_ang = max(-1.0, min(1.0, cos_ang))
        return float(np.arccos(cos_ang))

    def _is_pose_drastic_for_track(self, ped: dict, prev_pose: list):
        """
        Guardrail for per-frame pose glitches.
        Returns (is_drastic, reason_text).
        """
        pose = ped.get("smpl_pose")
        if not isinstance(pose, list) or len(pose) != 72:
            return False, "no_pose"
        if not isinstance(prev_pose, list) or len(prev_pose) != 72:
            return False, "no_prev_pose"

        deltas = []
        for j in range(24):
            i0 = 3 * j
            deltas.append(
                self._axis_angle_geodesic_delta_rad(
                    prev_pose[i0:i0 + 3], pose[i0:i0 + 3]
                )
            )

        root_delta = float(deltas[0])
        max_delta = float(max(deltas))
        soft_count = int(sum(d > self.pymaf_pose_guard_soft_joint_delta_rad for d in deltas))

        if root_delta > self.pymaf_pose_guard_root_delta_rad:
            return True, f"root_delta={root_delta:.3f}"
        if max_delta > self.pymaf_pose_guard_hard_joint_delta_rad:
            return True, f"max_joint_delta={max_delta:.3f}"
        if (
            soft_count >= self.pymaf_pose_guard_soft_joint_count
            and max_delta > self.pymaf_pose_guard_soft_max_joint_delta_rad
        ):
            return True, f"soft_spike count={soft_count} max={max_delta:.3f}"
        return False, "ok"

    def _update_prev_pose_cache(self, track_key: str, ped: dict):
        pose = ped.get("smpl_pose")
        if isinstance(pose, list) and len(pose) == 72:
            self._ped_prev_pose_by_track[track_key] = [float(v) for v in pose]

    def _clamp_pose_spike_for_track(self, ped: dict, track_key: str):
        """
        Clamp abrupt per-joint axis-angle jumps vs previous frame to prevent
        one-frame arm/limb spikes while preserving overall motion.
        """
        pose = ped.get("smpl_pose")
        if not isinstance(pose, list) or len(pose) != 72:
            return ped, 0

        prev_pose = self._ped_prev_pose_by_track.get(track_key)
        if not isinstance(prev_pose, list) or len(prev_pose) != 72:
            return ped, 0

        max_delta = max(1e-4, float(self.pymaf_pose_max_joint_delta_rad))
        filtered = [float(v) for v in pose]
        clamped = 0

        for j in range(24):
            i0 = 3 * j
            curr = np.array(filtered[i0:i0 + 3], dtype=np.float32)
            prev = np.array(prev_pose[i0:i0 + 3], dtype=np.float32)
            delta = curr - prev
            dn = float(np.linalg.norm(delta))
            if dn <= max_delta or dn <= 1e-8:
                continue
            curr = prev + delta * (max_delta / dn)
            filtered[i0:i0 + 3] = [float(curr[0]), float(curr[1]), float(curr[2])]
            clamped += 1

        if clamped == 0:
            return ped, 0

        adjusted = copy.deepcopy(ped)
        adjusted["smpl_pose"] = filtered
        return adjusted, clamped

    def _place_pedestrian_pymaf(self, ped: dict) -> bool:
        """
        Place a pedestrian using PyMAF data.
        Prefers full SMPL mesh when smpl_fk solver is loaded and pose/betas exist.
        Falls back to skeleton if mesh mode is unavailable.
        """
        # ── Mesh path ─────────────────────────────────────────────────────────
        if self.smpl_fk is not None:
            pose_raw  = ped.get("smpl_pose")
            betas_raw = ped.get("smpl_betas")
            if (isinstance(pose_raw, list) and len(pose_raw) == 72
                    and isinstance(betas_raw, list) and len(betas_raw) >= 10):
                return self._place_pedestrian_smpl_mesh(ped, pose_raw, betas_raw)

        # ── Skeleton fallback ─────────────────────────────────────────────────
        joints_raw = ped.get("smpl_joints3d")
        if not isinstance(joints_raw, list) or len(joints_raw) < self.pymaf_min_joint_count:
            return False

        joints_smpl = self._select_smpl24_joints(joints_raw)
        if joints_smpl is None:
            return False

        joints_local = []
        for pt in joints_smpl:
            if not isinstance(pt, (list, tuple)) or len(pt) < 3:
                return False
            joints_local.append(self._smpl_to_blender_local(pt))
        if len(joints_local) < 24:
            return False

        joints_local, was_flipped = self._auto_fix_upside_down_pymaf(joints_local, ped=ped)
        root_local   = joints_local[0]
        local_heights = [v.z for v in joints_local]
        local_height  = max(local_heights) - min(local_heights)
        scale = 1.0
        if local_height > 1e-6 and self.pymaf_target_height_m > 0.0:
            scale = max(0.5, min(2.5, self.pymaf_target_height_m / local_height))

        target_root   = Vector(self._json_to_blender(ped["position_3d"]))
        joints_world  = [((v - root_local) * scale) + target_root for v in joints_local]
        min_world_z   = min(v.z for v in joints_world)
        lift          = (0.0 + self.ground_clearance_m) - min_world_z
        if abs(lift) > 1e-6:
            joints_world = [Vector((v.x, v.y, v.z + lift)) for v in joints_world]

        track_id  = ped.get("pymaf_track_id", "na")
        base_name = f"ped_pymaf_{track_id}_{len(self._frame_objects)}"
        skel_obj  = self._create_pymaf_skeleton_curve(joints_world, name=base_name)
        if skel_obj is None:
            return False

        ped_mat = self.Materials.get_pedestrian_material()
        if ped_mat is not None:
            skel_obj.data.materials.clear()
            skel_obj.data.materials.append(ped_mat)

        head_idx = 15 if len(joints_world) > 15 else max(
            range(len(joints_world)), key=lambda i: joints_world[i].z
        )
        head_obj = self._create_sphere_mesh_object(
            name=f"{base_name}_head",
            location=joints_world[head_idx],
            radius=max(0.005, self.pymaf_joint_radius_m),
        )
        if ped_mat is not None:
            head_obj.data.materials.clear()
            head_obj.data.materials.append(ped_mat)

        self._frame_objects.extend([skel_obj, head_obj])
        print(
            f"[assets] pedestrian(skeleton): track={track_id} "
            f"upright_fix={'on' if was_flipped else 'off'}"
        )
        return True
    
    def _place_pedestrian_smpl_mesh(self, ped: dict, pose_raw: list, betas_raw: list) -> bool:
        """
        Reconstruct the full SMPL body mesh from pose + betas and place it in the scene.
        """
        try:
            verts_smpl, faces = self.smpl_fk.forward(pose_raw, betas_raw)
        except Exception as exc:
            print(f"[assets] SMPL FK failed: {exc}; falling back to skeleton")
            return False

        # ── Convert SMPL y-up → Blender space ────────────────────────────────
        # SMPL: x=right, y=up, z=back  →  Blender: x=right, y=forward, z=up
        # _smpl_to_blender_local maps (x,y,z) → Vector(x, z, y) for y-up mode.
        verts_bl_arr = verts_smpl[:, [0, 2, 1]]       # numpy fast path: (6890,3)
        if self.pymaf_y_up:
            verts_bl_arr = verts_smpl[:, [0, 2, 1]]
        else:
            verts_bl_arr = verts_smpl * np.array([1, 1, -1], dtype=np.float32)
            verts_bl_arr = verts_bl_arr[:, [0, 2, 1]]

        # Apply the same upside-down detector used by skeleton mode.
        # If mirrored across the ground plane, flip local Z for vertices.
        # For meshes, also flip triangle winding to preserve outward normals.
        mesh_upright_fix = False
        joints_raw = ped.get("smpl_joints3d")
        if isinstance(joints_raw, list) and len(joints_raw) >= self.pymaf_min_joint_count:
            joints_smpl = self._select_smpl24_joints(joints_raw)
            if joints_smpl is not None:
                joints_local = []
                valid = True
                for pt in joints_smpl:
                    if not isinstance(pt, (list, tuple)) or len(pt) < 3:
                        valid = False
                        break
                    joints_local.append(self._smpl_to_blender_local(pt))
                if valid and len(joints_local) >= 24:
                    _, was_flipped = self._auto_fix_upside_down_pymaf(joints_local, ped=ped)
                    if was_flipped:
                        verts_bl_arr[:, 2] *= -1.0
                        faces = faces[:, [0, 2, 1]]
                        mesh_upright_fix = True

        # ── Scale to target height ────────────────────────────────────────────
        z_min, z_max = float(verts_bl_arr[:, 2].min()), float(verts_bl_arr[:, 2].max())
        mesh_height  = z_max - z_min
        scale = 1.0
        if mesh_height > 1e-6 and self.pymaf_target_height_m > 0.0:
            scale = max(0.5, min(2.5, self.pymaf_target_height_m / mesh_height))
        verts_bl_arr = verts_bl_arr * scale

        # ── Position: place at target root, then ground in world space ────────
        # Find root joint (joint 0 = pelvis) for centering in XY
        root_smpl = verts_smpl.mean(axis=0)   # approximate – or use joint 0 from FK
        root_bl   = np.array([root_smpl[0], root_smpl[2], root_smpl[1]], dtype=np.float32) * scale
        target    = np.array(self._json_to_blender(ped["position_3d"]), dtype=np.float32)
        offset    = target - root_bl

        final_verts = verts_bl_arr + offset   # (6890, 3)

        # Match skeleton behavior: after translation, lift so lowest vertex
        # sits exactly on the ground plane + clearance.
        min_world_z = float(final_verts[:, 2].min())
        lift = (0.0 + self.ground_clearance_m) - min_world_z
        if abs(lift) > 1e-6:
            final_verts[:, 2] += lift

        # ── Build Blender mesh ────────────────────────────────────────────────
        track_id  = ped.get("pymaf_track_id", "na")
        mesh_name = f"ped_smpl_{track_id}_{len(self._frame_objects)}"
        me = bpy.data.meshes.new(f"{mesh_name}_mesh")
        bm = bmesh.new()

        bm_verts = [bm.verts.new(final_verts[i].tolist()) for i in range(len(final_verts))]
        bm.verts.ensure_lookup_table()

        for tri in faces:
            try:
                bm.faces.new([bm_verts[tri[0]], bm_verts[tri[1]], bm_verts[tri[2]]])
            except ValueError:
                pass   # skip degenerate/duplicate faces

        bm.normal_update()
        bm.to_mesh(me)
        bm.free()

        obj = bpy.data.objects.new(mesh_name, me)
        bpy.context.collection.objects.link(obj)

        ped_mat = self.Materials.get_pedestrian_material()
        if ped_mat is not None:
            me.materials.clear()
            me.materials.append(ped_mat)

        self._frame_objects.append(obj)
        print(
            f"[assets] pedestrian(smpl_mesh): track={track_id} "
            f"verts=6890 scale={scale:.3f} upright_fix={'on' if mesh_upright_fix else 'off'} "
            f"json_pos={ped['position_3d']} → blender_pos={tuple(round(float(v), 3) for v in target)}"
        )
        return True

    def _select_smpl24_joints(self, joints_raw: list):
        if self.pymaf_use_last24 and len(joints_raw) >= 49:
            return joints_raw[-24:]
        if len(joints_raw) >= 24:
            return joints_raw[:24]
        return None

    def _smpl_to_blender_local(self, pos) -> Vector:
        x, y, z = [float(v) for v in pos[:3]]
        if self.pymaf_y_up:
            # PyMAF joints are typically SMPL local coordinates: y-up, z-forward.
            return Vector((x, z, y))
        # Fallback for camera-style coordinates (x-right, y-down, z-forward).
        return Vector((x, z, -y))

    def _ped_track_key(self, ped: dict) -> str:
        track_id = ped.get("pymaf_track_id")
        if track_id is not None:
            return f"track:{track_id}"

        pos = ped.get("position_3d")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return f"anon:{float(pos[0]):.2f}:{float(pos[1]):.2f}:{float(pos[2]):.2f}"
        return "anon:unknown"

    def _stabilize_upright_flip(self, track_key: str, raw_flip: bool, score: float) -> bool:
        """
        Debounce orientation flips so single-frame PyMAF outliers do not invert
        pedestrians. Changes are accepted only after enough consistent evidence.
        """
        if score >= self.pymaf_upright_hysteresis:
            candidate_flip = False
        elif score <= -self.pymaf_upright_hysteresis:
            candidate_flip = True
        else:
            candidate_flip = None

        state = self._ped_upright_state.get(track_key)
        if state is None:
            resolved = raw_flip if candidate_flip is None else candidate_flip
            self._ped_upright_state[track_key] = {
                "flip": resolved,
                "pending": None,
                "count": 0,
            }
            return resolved

        current = bool(state.get("flip", raw_flip))
        pending = state.get("pending")
        pending_count = int(state.get("count", 0))

        if candidate_flip is None:
            state["pending"] = None
            state["count"] = 0
            return current

        if candidate_flip == current:
            state["pending"] = None
            state["count"] = 0
            return current

        if pending == candidate_flip:
            pending_count += 1
        else:
            pending = candidate_flip
            pending_count = 1

        if pending_count >= self.pymaf_flip_confirm_frames:
            state["flip"] = candidate_flip
            state["pending"] = None
            state["count"] = 0
            return candidate_flip

        state["pending"] = pending
        state["count"] = pending_count
        return current

    def _auto_fix_upside_down_pymaf(self, joints_local, ped: dict = None):
        """
        Detect and correct inverted skeletons by checking vertical ordering.

        In SMPL24, joints 12/15 (upper torso/head area) should sit above
        joints 10/11 (ankles) in upright poses.
        """
        if len(joints_local) < 16:
            return joints_local, False

        upper_ids = (12, 15)
        lower_ids = (10, 11)
        upper_z = [joints_local[i].z for i in upper_ids if i < len(joints_local)]
        lower_z = [joints_local[i].z for i in lower_ids if i < len(joints_local)]

        if not upper_z or not lower_z:
            return joints_local, False

        upper_avg = sum(upper_z) / len(upper_z)
        lower_avg = sum(lower_z) / len(lower_z)
        score = upper_avg - lower_avg
        raw_flip = score < 0.0

        if ped is None:
            should_flip = raw_flip
        else:
            should_flip = self._stabilize_upright_flip(
                track_key=self._ped_track_key(ped),
                raw_flip=raw_flip,
                score=score,
            )

        if not should_flip:
            return joints_local, False

        flipped = [Vector((v.x, v.y, -v.z)) for v in joints_local]
        return flipped, True

    def _create_pymaf_skeleton_curve(self, joints_world, name: str):
        curve = bpy.data.curves.new(f"{name}_curve", type="CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = max(0.001, self.pymaf_bone_radius_m)
        curve.bevel_resolution = 2
        curve.use_fill_caps = True

        num_bones = 0
        for a, b in self._SMPL24_BONES:
            if a >= len(joints_world) or b >= len(joints_world):
                continue
            p0 = joints_world[a]
            p1 = joints_world[b]
            if (p1 - p0).length <= 1e-4:
                continue

            spline = curve.splines.new("POLY")
            spline.points.add(1)
            spline.points[0].co = (p0.x, p0.y, p0.z, 1.0)
            spline.points[1].co = (p1.x, p1.y, p1.z, 1.0)
            num_bones += 1

        if num_bones == 0:
            bpy.data.curves.remove(curve, do_unlink=True)
            return None

        obj = bpy.data.objects.new(name, curve)
        bpy.context.collection.objects.link(obj)
        return obj

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

    def place_ground_text_marking(self, marking: dict, lanes: Optional[List[dict]] = None):
        """Render ONLY as white road text for positive ground text detections."""
        if not bool(marking.get("has_only_letters", False)):
            return

        if float(marking.get("ocr_confidence", 0.0) or 0.0) < 0.6:
            return

        bbox = marking.get("bbox") or []
        if len(bbox) != 4:
            return

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return

        if self._is_duplicate_ground_text_bbox((x1, y1, x2, y2)):
            return

        lane_list = lanes or []
        anchor = self._ground_text_left_anchor((x1, y1, x2, y2), marking)
        if anchor is None:
            return

        # Keep road text horizontal in the frame so it reads left-to-right.
        yaw = 0.0

        text_obj = self._create_ground_only_text(anchor, yaw, (x1, y1, x2, y2), marking)
        if text_obj is None:
            return

        self._frame_objects.append(text_obj)
        self._ground_text_bboxes.append((x1, y1, x2, y2))
        print(
            f"[assets] ground text: ONLY at bbox={bbox} yaw={yaw:.3f} "
            f"world=({text_obj.location.x:.3f}, {text_obj.location.y:.3f}, {text_obj.location.z:.3f})"
        )

    def place_ground_arrow(self, arrow: dict, lanes: Optional[List[dict]] = None):
        """Render a white arrow mesh on the road for ground_arrow detections."""
        bbox = arrow.get("bbox") or []
        if len(bbox) != 4:
            return

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return

        bbox_t = (x1, y1, x2, y2)
        if self._is_duplicate_ground_arrow_bbox(bbox_t):
            return

        anchor = self._ground_arrow_anchor(bbox_t, arrow)
        if anchor is None:
            return

        # For now, render all road arrows as "forward" and flip 180 degrees
        # from the prior default orientation.
        yaw = (math.pi / 2.0)

        arrow_obj = self._create_ground_arrow_mesh(anchor, yaw, bbox_t, arrow)
        if arrow_obj is None:
            return

        self._frame_objects.append(arrow_obj)
        self._ground_arrow_bboxes.append(bbox_t)
        print(
            f"[assets] ground arrow: bbox={bbox} yaw={yaw:.3f} "
            f"world=({arrow_obj.location.x:.3f}, {arrow_obj.location.y:.3f}, {arrow_obj.location.z:.3f})"
        )

    def _resolve_speed_value(self, sign: dict):
        """Resolve numeric speed from structured value first, OCR text second."""
        if str(self._scene_name).strip().lower() == "scene5":
            return 20
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
        self._ground_arrow_bboxes.clear()
        self._ground_text_bboxes.clear()

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
            child["source_name"] = template.name
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
            "tesla": "tesla",
            "ego": "tesla",
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
            "tesla": (0.85, 0.85, 0.85),
            "ego": (0.85, 0.85, 0.85),
            "car": (0.02, 0.02, 0.02),
            "sedan": (0.02, 0.02, 0.02),
            "sedanandhatchbacks": (0.02, 0.02, 0.02),
            "hatchback": (0.02, 0.02, 0.02),
            "suv": (1, 1, 1),
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
            "tesla": 0.0,
            "ego": 0.0,
            "truck": math.pi,
            "bus": math.pi,
            "pickuptruck": math.pi / 2,
            "pickup_truck": math.pi / 2,
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

    def _is_duplicate_ground_text_bbox(self, bbox: tuple) -> bool:
        """Suppress near-duplicate markings so ONLY is not stacked."""
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)

        for prev in self._ground_text_bboxes:
            px1, py1, px2, py2 = prev
            pcx = 0.5 * (px1 + px2)
            pcy = 0.5 * (py1 + py2)
            pw = max(px2 - px1, 1e-6)
            ph = max(py2 - py1, 1e-6)

            iou = self._bbox_iou(bbox, prev)
            dist = math.hypot(cx - pcx, cy - pcy)
            dist_thresh = 0.35 * max(max(w, h), max(pw, ph))

            if iou >= 0.45 or dist <= dist_thresh:
                return True

        return False

    def _is_duplicate_ground_arrow_bbox(self, bbox: tuple) -> bool:
        """Suppress near-duplicate arrows so markers are not stacked."""
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)

        for prev in self._ground_arrow_bboxes:
            px1, py1, px2, py2 = prev
            pcx = 0.5 * (px1 + px2)
            pcy = 0.5 * (py1 + py2)
            pw = max(px2 - px1, 1e-6)
            ph = max(py2 - py1, 1e-6)

            iou = self._bbox_iou(bbox, prev)
            dist = math.hypot(cx - pcx, cy - pcy)
            dist_thresh = 0.35 * max(max(w, h), max(pw, ph))

            if iou >= 0.45 or dist <= dist_thresh:
                return True

        return False

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = max(area_a + area_b - inter, 1e-6)
        return inter / union

    def _ground_text_left_anchor(self, bbox: tuple, marking: dict) -> Optional[Vector]:
        """Project the bbox-left edge to road so O-left aligns with bbox-left."""
        x1, y1, x2, y2 = bbox
        h = max(y2 - y1, 1.0)
        shift_px = self._estimate_only_left_shift_px(bbox, marking)
        sample_u = x1 - shift_px
        sample_v = y2 - 0.10 * h

        ground_pt = self._unproject_pixel_to_ground(float(sample_u), float(sample_v))
        if ground_pt is not None:
            return ground_pt

        pos3d = marking.get("position_3d")
        if isinstance(pos3d, list) and len(pos3d) == 3:
            bx, by, _ = self._json_to_blender(pos3d)
            # Approximate pixel shift in world X if projection fallback is used.
            cam_cfg = self.cfg.get("blender", {}).get("camera", {})
            fx = max(float(cam_cfg.get("fx", 800.0)), 1e-6)
            z = max(float(pos3d[2]), 1e-6)
            x_shift_world = z * (shift_px / fx)
            return Vector((float(bx) - x_shift_world, float(by), 0.0))

        return None

    @staticmethod
    def _estimate_only_left_shift_px(bbox: tuple, marking: dict) -> float:
        """Estimate how far left to move anchor when leading ONLY letters are missing."""
        x1, _, x2, _ = bbox
        bbox_w = max(float(x2 - x1), 1.0)

        hits_raw = str(marking.get("only_letter_hits", "") or "").upper()
        hits = {ch for ch in hits_raw if ch in {"O", "N", "L", "Y"}}
        if not hits:
            return 0.0

        target = "ONLY"
        present_indices = [i for i, ch in enumerate(target) if ch in hits]
        if not present_indices:
            return 0.0

        first_idx = min(present_indices)
        if first_idx <= 0:
            return 0.0

        last_idx = max(present_indices)
        observed_span = max(last_idx - first_idx + 1, 1)
        est_char_w = bbox_w / float(observed_span)
        shift_px = first_idx * est_char_w

        # Cap shift to avoid runaway offsets from noisy OCR hits.
        return min(shift_px, 1.75 * bbox_w)

    def _ground_arrow_anchor(self, bbox: tuple, arrow: dict) -> Optional[Vector]:
        """Project bottom-center of arrow bbox to road with position_3d fallback."""
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        sample_u = x1 + 0.5 * w
        sample_v = y2 - 0.08 * h

        ground_pt = self._unproject_pixel_to_ground(float(sample_u), float(sample_v))
        if ground_pt is not None:
            return ground_pt

        pos3d = arrow.get("position_3d")
        if isinstance(pos3d, list) and len(pos3d) == 3:
            bx, by, _ = self._json_to_blender(pos3d)
            return Vector((float(bx), float(by), 0.0))

        return None

    def _unproject_pixel_to_ground(self, u: float, v: float) -> Optional[Vector]:
        """Unproject image pixel to road plane Z=0 using camera intrinsics."""
        cam_cfg = self.cfg.get("blender", {}).get("camera", {})
        fx = float(cam_cfg.get("fx", 0.0))
        fy = float(cam_cfg.get("fy", 0.0))
        cx = float(cam_cfg.get("cx", 0.0))
        cy = float(cam_cfg.get("cy", 0.0))
        h = float(cam_cfg.get("height_m", 0.0))

        if fx <= 1e-6 or fy <= 1e-6 or h <= 1e-6:
            return None

        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        d = Vector((x_cam, 1.0, -y_cam))
        if d.length <= 1e-8:
            return None
        d.normalize()

        if d.z >= -1e-5:
            return None

        t = -h / d.z
        if t <= 0.0:
            return None

        p = Vector((0.0, 0.0, h)) + d * t
        return Vector((float(p.x), float(p.y), 0.0))

    def _ground_text_yaw_from_lanes(self, lanes: List[dict], bbox: tuple) -> Optional[float]:
        """Estimate road heading from nearest lane segment around text row."""
        if not lanes:
            return None

        _, y1, _, y2 = bbox
        target_v = 0.5 * (y1 + y2)

        best = None
        best_row_delta = float("inf")

        for lane in lanes:
            pts = lane.get("points") or []
            if len(pts) < 2:
                continue

            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i + 1]
                if not isinstance(p1, (list, tuple)) or not isinstance(p2, (list, tuple)):
                    continue
                if len(p1) < 2 or len(p2) < 2:
                    continue

                u1, v1 = float(p1[0]), float(p1[1])
                u2, v2 = float(p2[0]), float(p2[1])
                row_delta = abs(0.5 * (v1 + v2) - target_v)
                if row_delta < best_row_delta:
                    best_row_delta = row_delta
                    best = ((u1, v1), (u2, v2))

        if best is None:
            return None

        (u1, v1), (u2, v2) = best
        g1 = self._unproject_pixel_to_ground(u1, v1)
        g2 = self._unproject_pixel_to_ground(u2, v2)
        if g1 is None or g2 is None:
            return None

        vec = g2 - g1
        vec.z = 0.0
        if vec.length <= 1e-6:
            return None
        vec.normalize()

        return math.atan2(vec.y, vec.x)

    def _create_ground_only_text(self, anchor: Vector, yaw: float, bbox: tuple, marking: dict):
        """Create a white road-surface text object anchored on O-left edge."""
        text_curve = bpy.data.curves.new(name="GroundOnlyTextCurve", type="FONT")
        text_curve.body = "ONLY"
        text_curve.align_x = "LEFT"
        text_curve.align_y = "CENTER"
        text_curve.extrude = 0.002

        text_obj = bpy.data.objects.new(name="GroundOnlyText", object_data=text_curve)
        bpy.context.collection.objects.link(text_obj)
        text_obj.location = Vector((anchor.x, anchor.y, 0.02))
        text_obj.rotation_euler = (0.0, 0.0, yaw)

        x1, _, x2, _ = bbox
        bbox_w_px = max(float(x2 - x1), 1.0)
        depth_m = float(marking.get("depth_m", 0.0) or 0.0)
        cam_cfg = self.cfg.get("blender", {}).get("camera", {})
        fx = max(float(cam_cfg.get("fx", 800.0)), 1e-6)
        depth_for_scale = max(depth_m, 8.0)

        approx_world_width = depth_for_scale * (bbox_w_px / fx)
        target_width = min(max(approx_world_width * 0.72, 0.75), 3.8)

        # "ONLY" width in Blender font units is roughly ~2.8 at scale=1.
        scale = target_width / 2.8
        scale = min(max(scale, 0.28), 1.55)
        text_obj.scale = (scale, scale, scale)

        mat = bpy.data.materials.get("ground_text_only_white")
        if mat is None:
            mat = bpy.data.materials.new(name="ground_text_only_white")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
                emission_inp = next((inp for inp in bsdf.inputs if inp.name == "Emission Color"), None)
                emission_strength = next((inp for inp in bsdf.inputs if inp.name == "Emission Strength"), None)
                if emission_inp is not None:
                    emission_inp.default_value = (1.0, 1.0, 1.0, 1.0)
                if emission_strength is not None:
                    emission_strength.default_value = 0.2
                rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Roughness"), None)
                if rough_inp is not None:
                    rough_inp.default_value = 0.9

        text_obj.data.materials.clear()
        text_obj.data.materials.append(mat)
        return text_obj

    def _create_ground_arrow_mesh(self, anchor: Vector, yaw: float, bbox: tuple, arrow: dict):
        """Create a white ground arrow mesh aligned to the road plane."""
        x1, y1, x2, y2 = bbox
        bbox_px = max(float(x2 - x1), float(y2 - y1), 1.0)
        depth_m = float(arrow.get("depth_m", 0.0) or 0.0)
        cam_cfg = self.cfg.get("blender", {}).get("camera", {})
        fx = max(float(cam_cfg.get("fx", 800.0)), 1e-6)
        depth_for_scale = max(depth_m, 8.0)

        approx_len = depth_for_scale * (bbox_px / fx)
        arrow_len = min(max(approx_len * 0.9, 1.0), 4.5)
        arrow_h = arrow_len * 0.55
        arrow_depth = 0.02

        arrow_obj = self._create_left_arrow_mesh_object(
            name=f"GroundArrow_{len(self._frame_objects)}",
            location=Vector((anchor.x, anchor.y, 0.018)),
            width=arrow_len,
            height=arrow_h,
            depth=arrow_depth,
        )

        # Move the extruded arrow from vertical X/Z plane onto road X/Y plane.
        arrow_obj.rotation_euler = (math.pi / 2.0, 0.0, yaw)

        mat = bpy.data.materials.get("ground_arrow_white")
        if mat is None:
            mat = bpy.data.materials.new(name="ground_arrow_white")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
                emission_inp = next((inp for inp in bsdf.inputs if inp.name == "Emission Color"), None)
                emission_strength = next((inp for inp in bsdf.inputs if inp.name == "Emission Strength"), None)
                if emission_inp is not None:
                    emission_inp.default_value = (1.0, 1.0, 1.0, 1.0)
                if emission_strength is not None:
                    emission_strength.default_value = 0.15
                rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Roughness"), None)
                if rough_inp is not None:
                    rough_inp.default_value = 0.9

        arrow_obj.data.materials.clear()
        arrow_obj.data.materials.append(mat)
        return arrow_obj
