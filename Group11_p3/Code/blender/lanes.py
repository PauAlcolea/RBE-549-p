"""
blender/lanes.py
================
Creates lane geometry in the Blender scene from detection JSON polylines.

Each lane is rendered as a flat ribbon (narrow plane mesh) lying on the
estimated ground plane. Color and dash pattern are set via materials.

Approach:
  - Project the image-space lane polypoints through the camera to the
	ground plane (Z=0 in world space, i.e. the road surface).
  - Extrude the polyline into a flat strip using the lane_width from config.
  - Assign a material: white solid, white dashed, or yellow solid.
  - For dashed lanes: either use a UV-tiled texture or place multiple
	short segments with gaps.
"""

from typing import List, Optional

import numpy as np


class LaneRenderer:
	"""Manages lane geometry objects in the Blender scene.

	Usage
	-----
	renderer = LaneRenderer(cfg)
	renderer.draw_lane(lane_dict)
	renderer.clear()             # call before each new frame
	"""

	def __init__(self, cfg: dict):
		self.cfg = cfg
		self.lane_width = cfg["blender"]["style"]["lane_width"]
		self.cam_cfg = cfg["blender"]["camera"]

		self._lane_objects = []
		self._mat_cache = {}

		# Threshold below which we skip drawing lanes
		self._min_conf = float(
			cfg.get("perception", {})
			.get("lanes", {})
			.get("confidence", 0.0)
		)

	# ── Public API ────────────────────────────────────────────────────────

	def draw_lane(self, lane: dict):
		"""Create a lane ribbon object for one lane.

		Parameters
		----------
		lane : dict with keys "points" [[x,y],...], "color", "confidence", optional "style".
		"""

		points = lane.get("points") or []
		if len(points) < 2:
			return

		confidence = float(lane.get("confidence", 1.0))
		if confidence < self._min_conf:
			return

		# Unproject image points to ground plane
		ground_pts = []
		for pt in points:
			wp = self._unproject_to_ground(pt)
			if wp is not None:
				ground_pts.append(wp)

		if len(ground_pts) < 2:
			return

		lane_obj = self._build_ribbon(ground_pts, self.lane_width)
		if lane_obj is None:
			return

		color = lane.get("color", "white")
		mat = self._get_material(color)

		# Attach material
		if mat is not None:
			if lane_obj.data.materials:
				lane_obj.data.materials.clear()
			lane_obj.data.materials.append(mat)

		self._lane_objects.append(lane_obj)

	def clear(self):
		"""Remove all lane objects from the previous frame."""
		try:
			import bpy
		except ImportError:
			# Not running inside Blender; nothing to clear.
			self._lane_objects.clear()
			return

		for obj in self._lane_objects:
			if obj.name in bpy.data.objects:
				bpy.data.objects.remove(obj, do_unlink=True)
		self._lane_objects.clear()

	# ── Helpers ───────────────────────────────────────────────────────────

	def _unproject_to_ground(self, img_pt: List[float]) -> Optional[tuple]:
		"""Unproject an image-space point (u, v) to the world ground plane (Z=0).

		Uses a simple pinhole camera model with intrinsics from cfg["blender"]["camera"].
		The camera is assumed to be at (0, 0, height_m) looking forward along +Y.

		The ray direction in world coordinates is approximated as:
			d = [(u - cx) / fx, 1, -(v - cy) / fy]

		We then intersect this ray with the plane Z=0.
		Returns (world_x, world_y, 0.0) or None if the ray does not hit the ground.
		"""

		fx = float(self.cam_cfg["fx"])
		fy = float(self.cam_cfg["fy"])
		cx = float(self.cam_cfg["cx"])
		cy = float(self.cam_cfg["cy"])
		h = float(self.cam_cfg["height_m"])

		u, v = float(img_pt[0]), float(img_pt[1])

		x_cam = (u - cx) / fx
		y_cam = (v - cy) / fy

		# Approximate world ray (x, forward, -z)
		d = np.array([x_cam, 1.0, -y_cam], dtype=float)
		norm = np.linalg.norm(d)
		if norm == 0.0:
			return None
		d /= norm

		# Camera origin in world coords
		o = np.array([0.0, 0.0, h], dtype=float)

		# Intersect with Z=0 plane: o.z + t * d.z = 0 -> t = -o.z / d.z
		if d[2] >= -1e-5:
			# Ray does not point downwards towards the ground
			return None

		t = -o[2] / d[2]
		if t <= 0.0:
			return None

		p = o + t * d
		return float(p[0]), float(p[1]), 0.0

	def _build_ribbon(self, ground_pts: list, width: float):
		"""Create a flat ribbon mesh from a sequence of ground-plane points.

		Returns a Blender object or None if creation fails.
		"""

		try:
			import bpy
			import bmesh
			from mathutils import Vector
		except ImportError:
			# Not running inside Blender; cannot build geometry.
			return None

		if len(ground_pts) < 2:
			return None

		bm = bmesh.new()
		verts_left = []
		verts_right = []
		half_w = float(width) / 2.0

		n = len(ground_pts)
		for i, pt in enumerate(ground_pts):
			x, y, z = float(pt[0]), float(pt[1]), float(pt[2])

			if i == 0:
				x2, y2, _ = ground_pts[i + 1]
				dir_vec = Vector((x2 - x, y2 - y, 0.0))
			elif i == n - 1:
				x2, y2, _ = ground_pts[i - 1]
				dir_vec = Vector((x - x2, y - y2, 0.0))
			else:
				x_prev, y_prev, _ = ground_pts[i - 1]
				x_next, y_next, _ = ground_pts[i + 1]
				dir_vec = Vector((x_next - x_prev, y_next - y_prev, 0.0))

			if dir_vec.length == 0.0:
				dir_vec = Vector((0.0, 1.0, 0.0))
			else:
				dir_vec.normalize()

			# Perpendicular vector in the ground plane
			perp = Vector((dir_vec.y, -dir_vec.x, 0.0))

			center = Vector((x, y, z))
			left = center - half_w * perp
			right = center + half_w * perp

			v_l = bm.verts.new(left)
			v_r = bm.verts.new(right)
			verts_left.append(v_l)
			verts_right.append(v_r)

		bm.verts.ensure_lookup_table()

		for i in range(n - 1):
			v1 = verts_left[i]
			v2 = verts_right[i]
			v3 = verts_right[i + 1]
			v4 = verts_left[i + 1]
			try:
				bm.faces.new((v1, v2, v3, v4))
			except ValueError:
				# Face may already exist if inputs are degenerate
				continue

		mesh = bpy.data.meshes.new("Lane")
		bm.to_mesh(mesh)
		bm.free()

		obj = bpy.data.objects.new("Lane", mesh)
		bpy.context.collection.objects.link(obj)
		return obj

	def _get_material(self, color: str):
		"""Return (or create) a Blender material for the given lane color."""

		try:
			import bpy
		except ImportError:
			return None

		color_in = str(color).lower().strip()
		if "yellow" in color_in:
			lane_color = "yellow"
		else:
			lane_color = "white"

		key = f"lane_{lane_color}"

		if key in self._mat_cache:
			return self._mat_cache[key]

		if key in bpy.data.materials:
			mat = bpy.data.materials[key]
			self._mat_cache[key] = mat
			return mat

		style_cfg = self.cfg["blender"]["style"]
		rgb_key = f"lane_solid_{lane_color}"
		rgb = style_cfg.get(rgb_key, style_cfg.get("lane_solid_white", [1.0, 1.0, 1.0]))
		r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])

		mat = bpy.data.materials.new(name=key)
		mat.use_nodes = True
		nodes = mat.node_tree.nodes
		links = mat.node_tree.links

		for n in list(nodes):
			nodes.remove(n)

		bsdf = nodes.new("ShaderNodeBsdfPrincipled")
		bsdf.location = (0, -80)
		bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)

		rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Roughness"), None)
		if rough_inp is not None:
			rough_inp.default_value = 1.0

		spec_inp = next((inp for inp in bsdf.inputs if inp.name == "Specular"), None)
		if spec_inp is not None:
			spec_inp.default_value = 0.0

		emit = nodes.new("ShaderNodeEmission")
		emit.location = (0, 120)
		emit.inputs["Color"].default_value = (r, g, b, 1.0)
		emit.inputs["Strength"].default_value = 4.0 #if lane_color == "yellow" else 2.0

		out = nodes.new("ShaderNodeOutputMaterial")
		out.location = (360, 30)

		if lane_color == "yellow":
			# Pure emission keeps yellow vivid and avoids white clipping from additive shading.
			links.new(emit.outputs["Emission"], out.inputs["Surface"])
		else:
			add = nodes.new("ShaderNodeAddShader")
			add.location = (190, 30)
			links.new(bsdf.outputs["BSDF"], add.inputs[0])
			links.new(emit.outputs["Emission"], add.inputs[1])
			links.new(add.outputs["Shader"], out.inputs["Surface"])

		self._mat_cache[key] = mat
		return mat

