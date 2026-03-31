"""
utils/geometry.py
=================
Shared geometry math used by both perception (depth lifting) and
Blender (unprojection to ground plane, coordinate transforms).

Most functions are pure numpy. Camera calibration helpers import
OpenCV lazily so the rest of the module stays lightweight.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Container for checkerboard-based camera calibration outputs."""

    K: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms_error: float
    mean_reprojection_error: float
    used_images: list[str]
    total_images: int

    # these methods make it easier to use fx, fy, cx, and cy by renaming them
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def as_dict(self) -> dict:
        # this dictionary makes it easier to use all of the informationfrom this class
        width, height = self.image_size
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "image_width": int(width),
            "image_height": int(height),
            "rms_error": float(self.rms_error),
            "mean_reprojection_error": float(self.mean_reprojection_error),
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist(),
            "used_images": list(self.used_images),
            "total_images": int(self.total_images),
        }

# Square size will not affect the intrinsics, only the extrinsics scale, which is alright for us
def _checkerboard_object_points(
    pattern_size: tuple[int, int],
    square_size: float,
) -> np.ndarray:
    """Build the planar 3D checkerboard coordinates for OpenCV calibration."""
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= np.float32(square_size)
    return objp

# this is the main calibratin function that returns the CalibrationResult object
def calibrate_intrinsics_from_checkerboard(
    image_dir: str | Path,
    pattern_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    image_glob: str = "*.jpg",
) -> CalibrationResult:
    """
    Estimate camera intrinsics from a folder of checkerboard images.

    Parameters
    ----------
    image_dir : str | Path
        Folder containing calibration frames for a single camera.
    pattern_size : tuple[int, int], default=(9, 6)
        Number of inner checkerboard corners as (columns, rows).
    square_size : float, default=1.0
        Physical checker square size in any consistent unit. This does not
        change fx/fy/cx/cy, but it does affect the extrinsic scale.
    image_glob : str, default="*.jpg"
        Glob used to collect calibration images from ``image_dir``.

    Returns
    -------
    CalibrationResult
        Intrinsic matrix, distortion coefficients, and error metrics.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for calibration. Install opencv-python-headless "
            "or activate the project environment before calling this helper."
        ) from exc

    image_dir = Path(image_dir)
    image_paths = sorted(image_dir.glob(image_glob))
    if not image_paths:
        raise FileNotFoundError(
            f"No calibration images matching '{image_glob}' were found in {image_dir}."
        )

    objp = _checkerboard_object_points(pattern_size, square_size)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    used_images: list[str] = []
    image_size: tuple[int, int] | None = None

    corner_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )

    for image_path in image_paths:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(
                "All calibration images must have the same resolution. "
                f"Expected {image_size}, got {current_size} for {image_path.name}."
            )

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        if not found or corners is None:
            continue

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            corner_criteria,
        )

        objpoints.append(objp.copy())
        imgpoints.append(refined_corners)
        used_images.append(image_path.name)

    if image_size is None:
        raise RuntimeError(f"Unable to read any calibration images from {image_dir}.")
    if len(objpoints) < 3:
        raise RuntimeError(
            "Calibration needs at least 3 valid checkerboard detections. "
            f"Only found {len(objpoints)} in {len(image_paths)} images."
        )

    rms_error, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    per_view_errors = []
    for obj_pts, img_pts, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_coeffs)
        error = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(error))

    return CalibrationResult(
        K=K.astype(np.float64),
        dist_coeffs=dist_coeffs.astype(np.float64),
        image_size=image_size,
        rms_error=float(rms_error),
        mean_reprojection_error=float(np.mean(per_view_errors)),
        used_images=used_images,
        total_images=len(image_paths),
    )


def write_intrinsics_to_config(
    calibration: CalibrationResult,
    config_path: str | Path | None = None,
) -> Path:
    """
    Update ``blender.camera`` intrinsics in config.yaml while preserving comments.

    Only ``fx``, ``fy``, ``cx``, and ``cy`` are rewritten. After updating the
    YAML file, the config.json sidecar is refreshed to keep Blender reads in sync.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    replacements = {
        "fx": f"{calibration.fx:.6f}",
        "fy": f"{calibration.fy:.6f}",
        "cx": f"{calibration.cx:.6f}",
        "cy": f"{calibration.cy:.6f}",
    }
    updated_keys: set[str] = set()

    in_blender = False
    in_camera = False
    blender_indent = None
    camera_indent = None

    section_pattern = re.compile(r"^(\s*)([A-Za-z_][\w-]*)\s*:\s*(?:#.*)?$")
    value_pattern = re.compile(r"^(\s*)(fx|fy|cx|cy)(\s*:\s*)([^#\n]*?)(\s*(#.*)?)?(\r?\n)?$")

    for idx, line in enumerate(lines):
        stripped = line.strip()

        section_match = section_pattern.match(line)
        if section_match:
            indent = len(section_match.group(1))
            section_name = section_match.group(2)

            if in_camera and indent <= camera_indent and section_name != "camera":
                in_camera = False
                camera_indent = None
            if in_blender and indent <= blender_indent and section_name != "blender":
                in_blender = False
                blender_indent = None

            if section_name == "blender" and indent == 0:
                in_blender = True
                blender_indent = indent
                continue

            if in_blender and section_name == "camera":
                in_camera = True
                camera_indent = indent
                continue

        if not in_camera:
            continue

        value_match = value_pattern.match(line)
        if not value_match:
            continue

        key = value_match.group(2)
        if key not in replacements:
            continue

        suffix = value_match.group(5) or ""
        newline = value_match.group(7) or ""
        lines[idx] = (
            f"{value_match.group(1)}{key}{value_match.group(3)}"
            f"{replacements[key]}{suffix}{newline}"
        )
        updated_keys.add(key)

    missing = sorted(set(replacements) - updated_keys)
    if missing:
        raise RuntimeError(
            "Could not find all blender.camera intrinsic keys in config.yaml. "
            f"Missing: {missing}"
        )

    config_path.write_text("".join(lines), encoding="utf-8")

    from utils.io_utils import load_config

    load_config(str(config_path))
    return config_path


def calibrate_front_camera_intrinsics(
    calib_dir: str | Path | None = None,
    pattern_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    update_config: bool = False,
    config_path: str | Path | None = None,
) -> CalibrationResult:
    """
    Convenience wrapper for the repo's front-camera calibration images.

    By default this looks for ``Group11_p3/Data/Calib/front`` relative to this
    file, which matches the current project layout.
    """
    if calib_dir is None:
        calib_dir = Path(__file__).resolve().parents[2] / "Data" / "Calib" / "front" / "undistorted"

    result = calibrate_intrinsics_from_checkerboard(
        image_dir=calib_dir,
        pattern_size=pattern_size,
        square_size=square_size,
    )
    if update_config:
        write_intrinsics_to_config(result, config_path=config_path)
    return result


# # ── Camera model ──────────────────────────────────────────────────────────────

def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Build a 3×3 camera intrinsic matrix K from focal lengths and principal point.

    Returns
    -------
    K : np.ndarray, shape (3, 3)
    """
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)
