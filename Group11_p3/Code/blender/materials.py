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

    def get_pose_joint_material(self):
        """Return emissive material for pedestrian joints."""
        rgb = self.style.get("pose_joint_color", self.style["ped_color"])
        return self._get_or_create("pose_joint", rgb)

    def get_pose_bone_material(self):
        """Return emissive material for pedestrian skeleton links."""
        rgb = self.style.get("pose_bone_color", self.style["ped_color"])
        return self._get_or_create("pose_bone", rgb)

    def get_traffic_cone_material(self):
        """Return orange material for traffic-cone meshes."""
        return self._get_or_create_matte("traffic_cone_body", [1.0, 0.45, 0.0])

    def get_traffic_light_body_material(self):
        """Return yellow material for traffic-light housing/body."""
        return self._get_or_create_matte("traffic_light_body", [1.0, 0.9, 0.1])

    def get_trash_can_bin_material(self):
        """Return matte green material for trash-can bin body (#027c49)."""
        return self._get_or_create_matte("trash_can_bin", [2 / 255, 124 / 255, 73 / 255])

    def get_trash_can_lid_material(self):
        """Return matte black material for trash-can lid."""
        return self._get_or_create_matte("trash_can_lid", [0.03, 0.03, 0.03])

    def get_trash_can_wheels_material(self):
        """Return matte black material for trash-can wheels."""
        return self._get_or_create_matte("trash_can_wheels", [0.03, 0.03, 0.03])

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

        # Rebuild every time so pre-existing materials from imported assets
        # cannot keep stale node links or coordinate modes.
        for n in list(nodes):
            nodes.remove(n)

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (200, 0)
        out_node = nodes.new("ShaderNodeOutputMaterial")
        out_node.location = (400, 0)

        texcoord_node = nodes.new("ShaderNodeTexCoord")
        texcoord_node.location = (-220, 0)

        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.location = (0, 0)
        image = bpy.data.images.load(str(texture_path), check_existing=True)
        # Prevent white fringes from premultiplied-alpha interpretation.
        if hasattr(image, "alpha_mode"):
            image.alpha_mode = "STRAIGHT"
        tex_node.image = image
        # Some stop-sign UVs extend slightly outside [0, 1]. EXTEND avoids
        # hard clipping at the border in those cases.
        tex_node.extension = "EXTEND"

        # Sample by mesh UVs (not Generated coords) for stable mapping.
        links.new(texcoord_node.outputs["UV"], tex_node.inputs["Vector"])

        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        alpha_in = next((inp for inp in bsdf.inputs if inp.name == "Alpha"), None)
        if alpha_in is not None:
            links.new(tex_node.outputs["Alpha"], alpha_in)
        links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

        # Use alpha blending/shadow handling for textures with transparent
        # borders when supported by the current Blender version.
        if hasattr(mat, "blend_method"):
            mat.blend_method = "HASHED"
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = "HASHED"

        # Preserve existing slots (e.g., pole material) and replace only the
        # sign-face slot when possible.
        target_slot = 0
        for idx, existing_mat in enumerate(obj.data.materials):
            if existing_mat is None:
                continue
            name = existing_mat.name.lower()
            if any(token in name for token in ("sign", "stop", "face", "board")):
                target_slot = idx
                break

        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        elif target_slot < len(obj.data.materials):
            obj.data.materials[target_slot] = mat
        else:
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

    def _get_or_create_matte(self, key: str, rgb: list):
        """Return a cached non-emissive material for realistic object surfaces."""
        if key in self._cache:
            return self._cache[key]

        mat = bpy.data.materials.new(name=key)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for n in list(nodes):
            nodes.remove(n)

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)

        rough_inp = next((inp for inp in bsdf.inputs if inp.name == "Roughness"), None)
        if rough_inp is not None:
            rough_inp.default_value = 0.9

        spec_inp = next((inp for inp in bsdf.inputs if inp.name == "Specular"), None)
        if spec_inp is not None:
            spec_inp.default_value = 0.1

        out_node = nodes.new("ShaderNodeOutputMaterial")
        out_node.location = (220, 0)
        links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

        self._cache[key] = mat
        return mat
