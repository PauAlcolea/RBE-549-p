# """
# perception/lanes.py
# ===================
# Lane detection using CLRNet (or LaneATT as fallback).

# Outputs per frame:
#   - List of lane polylines (image coordinates)
#   - Each lane tagged with color ("white" | "yellow") and style ("solid" | "dashed")

# Color classification uses HSV thresholding on the lane region.
# Dashed vs solid is inferred from the density/continuity of detected points.
# """

# from dataclasses import dataclass, field
# from typing import List, Tuple
# import numpy as np
from pathlib import Path
from .Ultra_Fast_Lane_Detection_v2.deploy.trt_infer import UFLDv2
import tarfile
from Group11_p3.Code.utils.io_utils import download_file_if_missing


# @dataclass
# class Lane:
#     """A single detected lane line."""
#     points: List[Tuple[float, float]]   # [(x, y), ...] in image pixels, top to bottom
#     color: str = "white"                # "white" | "yellow"
#     style: str = "solid"                # "solid" | "dashed"
#     confidence: float = 1.0


class LaneDetector:
    """
    Wraps UFLD V2 for lane detection.

    Usage
    -----
    detector = LaneDetector(cfg, device="cuda")
    lanes = detector.detect(frame_bgr)   # list[Lane]
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.model_name = cfg["perception"]["lanes"]["model"]
        self.max_lanes  = cfg["perception"]["lanes"]["max_lanes"]
        self.model = self._load_model(cfg["weights"]["lanes"])

    def _load_model(self, weights_path: str = None):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            resources_url = self.cfg["perception"]["anes"].get("resources_url")
            resources_dir = Path("weights") / "ufld_resources"
            download_file_if_missing(Path("weights") / "ufld_resources.tar.gz", resources_url)
            resources_dir.mkdir(parents=True, exist_ok=True)
            tar_path = Path("weights") / "ufld_resources.tar.gz"
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(resources_dir)

            # 3. Optionally delete the archive
            tar_path.unlink(missing_ok=True)

        engine_path = weights_path / "culane_res34.engine"
        config_path = Path(__file__).parent / "Ultra_Fast_Lane_Detection_v2" / "configs" / "culane_res34.py"
        ori_size = (1280, 960)
        model = UFLDv2(engine_path, config_path, ori_size)
        if model is not None:
            print(f"[LaneDetector] Loaded UFLD V2 model from {engine_path}")
        return model

#     def detect(self, frame_bgr: np.ndarray) -> List[Lane]:
#         """
#         Run lane detection on one BGR frame.

#         Returns
#         -------
#         list[Lane]
#             Detected lanes with points in image coordinates.
#         """
#         if self.model is None:
#             return []

#         # TODO: implement
#         # 1. Preprocess frame (resize, normalize per CLRNet requirements)
#         # 2. Run model forward pass
#         # 3. Post-process to get polyline coordinates
#         # 4. Classify each lane's color and style (call helpers below)
#         raise NotImplementedError("LaneDetector.detect not yet implemented")

#     # ── Helpers ───────────────────────────────────────────────────────────

#     def _classify_color(self, frame_bgr: np.ndarray, points: list) -> str:
#         """
#         Sample pixels along the lane and classify as white or yellow
#         using HSV thresholding.

#         Returns "white" or "yellow".
#         """
#         # TODO: implement
#         # import cv2
#         # Sample ~10 points along the lane, extract small patches,
#         # convert to HSV, check if hue falls in yellow range (~15-35 deg).
#         # If majority of samples are yellow → "yellow", else "white".
#         raise NotImplementedError

#     def _classify_style(self, points: list, img_height: int) -> str:
#         """
#         Infer dashed vs solid from the vertical distribution of detected points.
#         Gaps between point clusters indicate dashes.

#         Returns "solid" or "dashed".
#         """
#         # TODO: implement
#         # Sort points by y. Compute gaps between consecutive y values.
#         # If max gap > threshold → "dashed", else "solid".
#         raise NotImplementedError
