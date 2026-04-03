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
import math
import bpy

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

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
        self._speed_limit_cache_dir = None
        self._speed_limit_template_png = None

    def get_vehicle_material(self):
        """Return flat emissive material for car silhouettes."""
        return self._get_or_create("vehicle", self.style["car_color"])

    def get_pedestrian_material(self):
        """Return flat emissive material for pedestrian silhouettes."""
        return self._get_or_create("pedestrian", self.style["ped_color"])

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

        mapping_node = nodes.new("ShaderNodeMapping")
        mapping_node.location = (-90, 0)

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

        # Optional UV alignment controls for stop-sign texture.
        # Can be overridden in config under blender.style.stop_sign_uv.
        uv_cfg = (
            self.cfg.get("blender", {})
            .get("style", {})
            .get("stop_sign_uv", {})
        )
        stem = texture_path.stem.lower()
        if stem == "stopsignimage":
            u_offset = float(uv_cfg.get("u_offset", 0.0))
            v_offset = float(uv_cfg.get("v_offset", 0.0))
            rot_deg = float(uv_cfg.get("rotation_deg", 0.0))
            u_scale = float(uv_cfg.get("u_scale", 1.0))
            v_scale = float(uv_cfg.get("v_scale", 0.92))
        elif stem.startswith("speed_limit_"):
            speed_uv_cfg = (
                self.cfg.get("blender", {})
                .get("style", {})
                .get("speed_limit_uv", {})
            )
            u_offset = float(speed_uv_cfg.get("u_offset", 0.0))
            v_offset = float(speed_uv_cfg.get("v_offset", 0.0))
            rot_deg = float(speed_uv_cfg.get("rotation_deg", 0.0))
            u_scale = float(speed_uv_cfg.get("u_scale", 1.0))
            v_scale = float(speed_uv_cfg.get("v_scale", 1.0))
        else:
            u_offset, v_offset = 0.0, 0.0
            rot_deg = 0.0
            u_scale, v_scale = 1.0, 1.0

        mapping_node.inputs["Location"].default_value = (u_offset, v_offset, 0.0)
        mapping_node.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rot_deg))
        mapping_node.inputs["Scale"].default_value = (u_scale, v_scale, 1.0)

        # Sample by mesh UVs (not Generated coords) for stable mapping.
        links.new(texcoord_node.outputs["UV"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], tex_node.inputs["Vector"])

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

    def get_speed_limit_texture(self, speed_value=None):
        """Return a PNG texture path for a speed-limit sign.

        Uses Data/Assets/Speed_Limit_blank_sign.png as the template and
        overlays centered speed_value text when provided.
        """
        template_path = self._resolve_speed_limit_template_png()
        if template_path is None:
            return None

        speed_num = self._coerce_speed_value(speed_value)
        if speed_num is None:
            return template_path

        if Image is None or ImageDraw is None or ImageFont is None:
            return template_path

        cache_dir = self._get_speed_limit_cache_dir()
        out_path = cache_dir / f"speed_limit_{speed_num}.png"
        if out_path.exists():
            return out_path

        try:
            img = Image.open(template_path).convert("RGBA")
            draw = ImageDraw.Draw(img)

            txt_cfg = (
                self.cfg.get("blender", {})
                .get("style", {})
                .get("speed_limit_text", {})
            )

            text = str(speed_num)
            font_scale = float(txt_cfg.get("font_scale", 0.40))
            y_ratio = float(txt_cfg.get("y_ratio", 0.69))
            text_color = tuple(int(c) for c in txt_cfg.get("text_color", [0, 0, 0]))
            stroke_width = int(txt_cfg.get("stroke_width", 4))
            stroke_color = tuple(int(c) for c in txt_cfg.get("stroke_color", [255, 255, 255]))

            target_size = max(24, int(min(img.width, img.height) * font_scale))
            print(f"[materials] Generating speed limit texture for {speed_num} mph with font size {target_size}px")
            font = self._load_speed_limit_font(target_size)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = (img.width - text_w) * 0.5
            text_y = img.height * y_ratio - (text_h * 0.5)

            draw.text(
                (text_x, text_y),
                text,
                fill=(*text_color, 255),
                font=font,
                stroke_width=max(0, stroke_width),
                stroke_fill=(*stroke_color, 255),
            )
            img.save(out_path)
            return out_path
        except Exception as exc:
            print(f"[materials] WARNING: failed to generate speed sign texture: {exc}")
            return template_path

    @staticmethod
    def pillow_available():
        """Return whether PIL text overlay is available in Blender Python."""
        return Image is not None and ImageDraw is not None and ImageFont is not None

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

    def _get_speed_limit_cache_dir(self):
        """Return writable cache folder for generated speed-sign textures."""
        if self._speed_limit_cache_dir is not None:
            return self._speed_limit_cache_dir

        frames_dir = Path(self.cfg.get("paths", {}).get("frames_dir", "../Outputs/Frames"))
        cache_dir = frames_dir.parent / "etc" / "speed_limit_textures"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._speed_limit_cache_dir = cache_dir
        return cache_dir

    def _resolve_speed_limit_template_png(self):
        """Resolve Speed_Limit_blank_sign.png from the assets directory."""
        if self._speed_limit_template_png is not None and self._speed_limit_template_png.exists():
            return self._speed_limit_template_png

        assets_dir = Path(self.cfg.get("paths", {}).get("assets_dir", "../Data/Assets"))
        png_path = assets_dir / "Speed_Limit_blank_sign.png"

        if png_path.exists():
            self._speed_limit_template_png = png_path
            return png_path

        print(f"[materials] WARNING: missing speed-limit template: {png_path}")
        return None

    @staticmethod
    def _coerce_speed_value(speed_value):
        if speed_value is None:
            return None
        try:
            value = int(float(speed_value))
        except Exception:
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _load_speed_limit_font(size: int):
        if ImageFont is None:
            return None

        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Impact.ttf",
        ]

        for font_path in candidates:
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue

        return ImageFont.load_default()

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
