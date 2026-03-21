# """
# blender/materials.py
# ====================
# Creates and manages Blender materials for all scene objects.

# Responsibilities:
#   - Tesla-style flat/emissive materials for car silhouettes, pedestrians
#   - Traffic light color switching (red/yellow/green emission)
#   - Stop sign texture application
#   - Lane materials (delegated from lanes.py, shared cache here)
# """
# from pathlib import Path


# class MaterialLibrary:
#     """
#     Centralized material cache. All materials are created once and reused.

#     Usage
#     -----
#     mats = MaterialLibrary(cfg)
#     mats.apply_texture(stop_sign_obj, texture_path)
#     mats.set_traffic_light_color("red")
#     """

#     def __init__(self, cfg: dict):
#         self.cfg = cfg
#         self.style = cfg["blender"]["style"]
#         self._cache = {}    # key → bpy.types.Material
#         self._traffic_light_mat = None   # reference for fast color swap

#     def get_vehicle_material(self):
#         """Return flat emissive material for car silhouettes."""
#         return self._get_or_create("vehicle", self.style["car_color"])

#     def get_pedestrian_material(self):
#         """Return flat emissive material for pedestrian silhouettes."""
#         return self._get_or_create("pedestrian", self.style["ped_color"])

#     def get_lane_material(self, color: str, style: str):
#         """
#         Return material for a lane stripe.
#         color: "white" | "yellow"
#         style: "solid" | "dashed"
#         """
#         key = f"lane_{color}_{style}"
#         rgb = self.style.get(f"lane_{style}_{color}",
#                              self.style["lane_solid_white"])
#         return self._get_or_create(key, rgb)

#     def apply_texture(self, obj, texture_path: Path):
#         """
#         Apply an image texture to a Blender object (used for stop sign).
#         Assumes the object already has UV coordinates.
#         """
#         # TODO: implement
#         # import bpy
#         # mat = bpy.data.materials.new(name="stop_sign_tex")
#         # mat.use_nodes = True
#         # nodes = mat.node_tree.nodes
#         # links = mat.node_tree.links
#         # bsdf = nodes["Principled BSDF"]
#         # tex_node = nodes.new("ShaderNodeTexImage")
#         # tex_node.image = bpy.data.images.load(str(texture_path))
#         # links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
#         # obj.data.materials.clear()
#         # obj.data.materials.append(mat)
#         pass

#     def set_traffic_light_color(self, color: str):
#         """
#         Update the traffic light asset's emission color for this frame.
#         color: "red" | "yellow" | "green" | "unknown"

#         Called once per frame after placing traffic light assets.
#         """
#         color_map = {
#             "red":     self.style["light_red"],
#             "yellow":  self.style["light_yellow"],
#             "green":   self.style["light_green"],
#             "unknown": [0.3, 0.3, 0.3],
#         }
#         rgb = color_map.get(color, color_map["unknown"])

#         # TODO: implement
#         # if self._traffic_light_mat is None:
#         #     self._traffic_light_mat = self._get_or_create("traffic_light", rgb)
#         # bsdf = self._traffic_light_mat.node_tree.nodes["Principled BSDF"]
#         # bsdf.inputs["Emission"].default_value = (*rgb, 1.0)
#         pass

#     # ── Helpers ───────────────────────────────────────────────────────────

#     def _get_or_create(self, key: str, rgb: list):
#         """
#         Return a cached material or create a new flat emissive one.
#         rgb is a list of 3 floats in [0, 1].
#         """
#         if key in self._cache:
#             return self._cache[key]

#         # TODO: implement
#         # import bpy
#         # mat = bpy.data.materials.new(name=key)
#         # mat.use_nodes = True
#         # bsdf = mat.node_tree.nodes["Principled BSDF"]
#         # bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
#         # bsdf.inputs["Roughness"].default_value = 1.0
#         # bsdf.inputs["Specular"].default_value = 0.0
#         # bsdf.inputs["Emission"].default_value = (*rgb, 1.0)
#         # bsdf.inputs["Emission Strength"].default_value = 1.5  # slight glow
#         # self._cache[key] = mat
#         # return mat
#         raise NotImplementedError(f"_get_or_create not yet implemented (key={key})")
