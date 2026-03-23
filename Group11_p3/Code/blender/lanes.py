# """
# blender/lanes.py
# ================
# Creates lane geometry in the Blender scene from detection JSON polylines.

# Each lane is rendered as a flat ribbon (narrow plane mesh) lying on the
# estimated ground plane. Color and dash pattern are set via materials.

# Approach:
#   - Project the image-space lane polypoints through the camera to the
#     ground plane (Y=0 in world space, i.e. the road surface).
#   - Extrude the polyline into a flat strip using the lane_width from config.
#   - Assign a material: white solid, white dashed, or yellow solid.
#   - For dashed lanes: either use a UV-tiled texture or place multiple
#     short segments with gaps.
# """
# from typing import List


# class LaneRenderer:
#     """
#     Manages lane geometry objects in the Blender scene.

#     Usage
#     -----
#     renderer = LaneRenderer(cfg)
#     renderer.draw_lane(lane_dict)
#     renderer.clear()             # call before each new frame
#     """

#     def __init__(self, cfg: dict):
#         self.cfg = cfg
#         self.lane_width = cfg["blender"]["style"]["lane_width"]
#         self.cam_cfg    = cfg["blender"]["camera"]
#         self._lane_objects = []

#     def draw_lane(self, lane: dict):
#         """
#         Create a lane ribbon object for one lane.

#         Parameters
#         ----------
#         lane : dict with keys "points" [[x,y],...], "color", "style"
#         """
#         # TODO: implement
#         # 1. Unproject image points to ground plane using camera intrinsics
#         #    ground_pts = [self._unproject_to_ground(pt) for pt in lane["points"]]
#         # 2. Build a ribbon mesh along the polyline
#         #    mesh = self._build_ribbon(ground_pts, self.lane_width)
#         # 3. Apply appropriate material
#         #    mat = self._get_material(lane["color"], lane["style"])
#         #    mesh.data.materials.append(mat)
#         # 4. Track for cleanup
#         #    self._lane_objects.append(mesh)
#         pass

#     def clear(self):
#         """Remove all lane objects from the previous frame."""
#         # TODO: implement
#         # import bpy
#         # for obj in self._lane_objects:
#         #     bpy.data.objects.remove(obj, do_unlink=True)
#         self._lane_objects.clear()

#     # ── Helpers ───────────────────────────────────────────────────────────

#     def _unproject_to_ground(self, img_pt: List[float]) -> tuple:
#         """
#         Unproject an image-space point (x, y) to the world ground plane (Z=0).

#         Uses the pinhole camera model + known camera height above ground.
#         Formula:
#           ray_dir = K_inv @ [u, v, 1]
#           t = -cam_height / ray_dir.y
#           world_pt = cam_pos + t * ray_dir

#         Returns (world_x, world_y, 0.0).
#         """
#         # TODO: implement
#         # fx = self.cam_cfg["fx"]
#         # fy = self.cam_cfg["fy"]
#         # cx = self.cam_cfg["cx"]
#         # cy = self.cam_cfg["cy"]
#         # h  = self.cam_cfg["height_m"]
#         # u, v = img_pt
#         # ray = np.array([(u - cx)/fx, (v - cy)/fy, 1.0])
#         # t = h / ray[1]   # intersect with Y = 0 plane
#         # world = t * ray
#         # return (world[0], world[2], 0.0)  # Blender coords
#         raise NotImplementedError

#     def _build_ribbon(self, ground_pts: list, width: float):
#         """
#         Create a flat ribbon mesh from a sequence of ground-plane points.
#         Returns a Blender object.
#         """
#         # TODO: implement using bmesh
#         # import bpy, bmesh, mathutils
#         # bm = bmesh.new()
#         # verts_left  = []
#         # verts_right = []
#         # for i, pt in enumerate(ground_pts):
#         #     ... compute left/right offset perpendicular to lane direction ...
#         #     verts_left.append(bm.verts.new(left))
#         #     verts_right.append(bm.verts.new(right))
#         # for i in range(len(ground_pts) - 1):
#         #     bm.faces.new([verts_left[i], verts_right[i], verts_right[i+1], verts_left[i+1]])
#         # mesh = bpy.data.meshes.new("lane")
#         # bm.to_mesh(mesh)
#         # obj = bpy.data.objects.new("lane", mesh)
#         # bpy.context.collection.objects.link(obj)
#         # return obj
#         raise NotImplementedError

#     def _get_material(self, color: str, style: str):
#         """
#         Return (or create) a Blender material for the given lane color and style.
#         Materials are cached by (color, style) to avoid duplicates.
#         """
#         # TODO: implement
#         # import bpy
#         # key = f"lane_{color}_{style}"
#         # if key in bpy.data.materials:
#         #     return bpy.data.materials[key]
#         # mat = bpy.data.materials.new(name=key)
#         # mat.use_nodes = True
#         # bsdf = mat.node_tree.nodes["Principled BSDF"]
#         # rgb = self.cfg["blender"]["style"][f"lane_{style}_{color}"]
#         # bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
#         # bsdf.inputs["Emission"].default_value = (*rgb, 1.0)   # unlit look
#         # bsdf.inputs["Emission Strength"].default_value = 2.0
#         # return mat
#         raise NotImplementedError
