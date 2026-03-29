"""
blender/materials.py
====================
Creates and manages Blender materials for all scene objects.

Responsibilities:
    - Tesla-style flat/emissive materials for car silhouettes, pedestrians
    - Traffic light color switching (red/yellow/green emission)
    - Stop sign texture application
    - Lane materials (delegated from lanes.py, shared cache here)
"""

from pathlib import Path
import bpy


class MaterialLibrary:
    """
    Centralized material cache. All materials are created once and reused.

    Usage
    -----
    mats = MaterialLibrary(cfg)
    mats.apply_texture(stop_sign_obj, texture_path)
    mats.set_traffic_light_color("red")
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.style = cfg["blender"]["style"]
        self._cache = {}  # key → bpy.types.Material
        self._traffic_light_mat = None  # reference for fast color swap

    def get_vehicle_material(self):
        """Return flat emissive material for car silhouettes."""
        return self._get_or_create("vehicle", self.style["car_color"])

    def get_pedestrian_material(self):
        """Return flat emissive material for pedestrian silhouettes."""
        return self._get_or_create("pedestrian", self.style["ped_color"])

    def get_lane_material(self, color: str, style: str):
        """
        Return material for a lane stripe.
        color: "white" | "yellow"
        style: "solid" | "dashed"
        """
        key = f"lane_{color}_{style}"
        rgb = self.style.get(f"lane_{style}_{color}", self.style["lane_solid_white"])
        return self._get_or_create(key, rgb)

    def apply_texture(self, obj, texture_path: Path):
        """Apply an image texture to a Blender object (used for stop sign).

        Assumes the object already has UV coordinates.
        Creates or reuses a simple Principled BSDF material with the image
        plugged into Base Color.
        """

        mat_name = f"tex_{texture_path.stem}"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Clear default nodes to have a predictable graph
            for n in list(nodes):
                nodes.remove(n)

            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (200, 0)
            out_node = nodes.new("ShaderNodeOutputMaterial")
            out_node.location = (400, 0)

            tex_node = nodes.new("ShaderNodeTexImage")
            tex_node.location = (0, 0)
            tex_node.image = bpy.data.images.load(str(texture_path))

            links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

        obj.data.materials.clear()
        obj.data.materials.append(mat)

    def set_traffic_light_color(self, color: str):
        """
        Legacy helper to update the traffic light material color.

        In the current implementation this simply ensures that a colored
        emission material for the given state exists via
        get_traffic_light_material().
        color: "red" | "yellow" | "green" | "unknown"

        Called once per frame after placing traffic light assets.
        """
        # Ensure the correct emission material exists; actual assignment to
        # the object is handled in AssetLibrary.place_traffic_light().
        self.get_traffic_light_material(color)

    def get_traffic_light_material(self, color: str):
        """Return a material for a traffic light with the given state color.

        Ensures the Base Color matches the light color so it is visible even
        on Blender versions where the Principled BSDF has no Emission input.
        """

        color_map = {
            "red": self.style["light_red"],
            "yellow": self.style["light_yellow"],
            "green": self.style["light_green"],
            "unknown": [0.3, 0.3, 0.3],
        }
        rgb = color_map.get(color, color_map["unknown"])

        # Use a dedicated emission material so the light is clearly visible,
        # independent of Principled BSDF input changes across Blender
        # versions.
        mat_name = f"traffic_light_{color}"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Clear any default nodes
            for n in list(nodes):
                nodes.remove(n)

            emit = nodes.new("ShaderNodeEmission")
            emit.location = (0, 0)
            emit.inputs["Color"].default_value = (*rgb, 1.0)
            emit.inputs["Strength"].default_value = 5.0

            out_node = nodes.new("ShaderNodeOutputMaterial")
            out_node.location = (200, 0)

            links.new(emit.outputs["Emission"], out_node.inputs["Surface"])

        self._traffic_light_mat = mat
        return mat

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_or_create(self, key: str, rgb: list):
        """
        Return a cached material or create a new flat emissive one.
        rgb is a list of 3 floats in [0, 1].
        """
        if key in self._cache:
            return self._cache[key]

        mat = bpy.data.materials.new(name=key)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")

        # Base color
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)

        # Roughness (if present) for a flat look
        rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Roughness"), None)
        if rough_inp is not None:
            rough_inp.default_value = 1.0

        # Specular (if present) to avoid shine
        spec_inp = next((inp for inp in bsdf.inputs if inp.name == "Specular"), None)
        if spec_inp is not None:
            spec_inp.default_value = 0.0

        # Emission (if present) for a slight glow
        emit_inp = next((inp for inp in bsdf.inputs if inp.name == "Emission"), None)
        if emit_inp is not None:
            emit_inp.default_value = (*rgb, 1.0)

        strength_inp = next(
            (inp for inp in bsdf.inputs if inp.name in ("Emission Strength",)), None
        )
        if strength_inp is not None:
            strength_inp.default_value = 1.5

        self._cache[key] = mat
        return mat
