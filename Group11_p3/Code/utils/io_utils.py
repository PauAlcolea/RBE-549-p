"""
utils/io_utils.py
=================
File I/O helpers shared by both the perception pipeline and the Blender module.

Covers:
  - Config loading (YAML)
  - Video frame extraction
  - Detection JSON read/write
  - Frame listing utilities

Data layout (under Data/Sequences/):
  scene1/
    Raw/
      2023-02-14_11-04-07-front.mp4
      2023-02-14_11-04-07-back.mp4
      2023-02-14_11-04-07-left_repeater.mp4
      2023-02-14_11-04-07-right_repeater.mp4
    Undist/
      2023-02-14_11-04-07-front_undistort.mp4
      2023-02-14_11-04-07-back_undistort.mp4
      2023-02-14_11-04-07-left_repeater_undistort.mp4
      2023-02-14_11-04-07-right_repeater_undistort.mp4
  scene2/ ...
"""

from pathlib import Path
from typing import Iterator, List, Tuple
import os
import numpy as np
import json


# # ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """
    Load config.yaml. Searches for it relative to the Code/ directory
    if no explicit path is given.

    Returns
    -------
    dict : parsed config
    """
    import yaml

    # if config_path is None:
    #     # Walk up from this file until we find config.yaml
    #     here = Path(__file__).resolve().parent
    #     for candidate in [here, here.parent, here.parent.parent]:
    #         p = candidate / "config.yaml"
    #         if p.exists():
    #             config_path = p
    #             break
    #     else:
    #         raise FileNotFoundError("config.yaml not found. Pass an explicit path.")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_file_if_missing(path: os.PathLike, url: str, timeout: float = 60.0) -> Path:
    """Download a file from ``url`` to ``path`` if it does not already exist.

    Uses a temporary file and atomic rename to avoid leaving corrupted files
    on failed downloads.
    """
    target = Path(path)
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    import urllib.request
    import shutil

    tmp_path = target.with_suffix(target.suffix + ".tmp")

    print(f"[io_utils] Downloading {url} -> {target} ...")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        tmp_path.replace(target)
        print(f"[io_utils] Download complete: {target}")
    except Exception as e:  # pragma: no cover - network dependent
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(f"Failed to download file from {url} to {target}: {e}") from e

    return target


# # ── Video frame extraction ────────────────────────────────────────────────────

def get_video_frames(
    scene_dir: Path,
    camera: str = "front",
    frame_skip: int = 1,
) -> List[Tuple[int, "np.ndarray"]]:
    """
    Extract all frames from the undistorted video of one camera in a scene.
    
    :param      scene_dir  : path to e.g. Data/Sequences/scene1/
    :param      camera     : which camera view to load ("front" | "back" | "left_repeater" | "right_repeater")
    :param      frame_skip : keep every Nth frame (1 = every frame)

    :retur list of (frame_idx, bgr_array) tuples
    """
    # import cv2

    # video_path = _find_video(scene_dir, camera)
    # print(f"[io_utils] Loading {video_path.name}")

    # cap = cv2.VideoCapture(str(video_path))
    # if not cap.isOpened():
    #     raise RuntimeError(f"cv2 could not open {video_path}")

    # frames = []
    # frame_idx = 0

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     if frame_idx % frame_skip == 0:
    #         frames.append((frame_idx, frame))
    #     frame_idx += 1

    # cap.release()
    # print(f"[io_utils] Loaded {len(frames)} frames (frame_skip={frame_skip})")
    # return frames
    return


def frame_generator(
    scene_dir: Path,
    camera: str = "front",
    frame_skip: int = 1,
) -> Iterator[Tuple[int, "np.ndarray"]]:
    """
    Memory-efficient generator version of get_video_frames.
    Yields (frame_idx, bgr_array) one at a time — prefer this for long sequences.

    Parameters
    ----------
    scene_dir  : path to e.g. Data/Sequences/scene1/
    camera     : which camera view to load
    frame_skip : yield every Nth frame
    """
    import cv2

    video_path = _find_video(scene_dir, camera)
    print(f"[io_utils] Streaming {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def _find_video(scene_dir: Path, camera: str = "front") -> Path:
    """
    Locate the undistorted video for a given camera inside a scene directory.
    Looks inside scene_dir/Undist/ for a file matching *-{camera}_undistort.mp4.
    The date-timestamp prefix varies per scene, so we glob for the pattern.

    :param      scene_dir : path to e.g. Data/Sequences/scene1/
    :param      camera    : one of "front" | "back" | "left_repeater" | "right_repeater"

    :returns    Path to the matched .mp4 file
    """
    undist_dir = Path(scene_dir) / "Undist"
    if not undist_dir.exists():
        raise FileNotFoundError(f"Undist/ directory not found in {scene_dir}")

    matches = list(undist_dir.glob(f"*-{camera}_undistort.mp4"))
    if not matches:
        raise FileNotFoundError(
            f"No undistorted video for camera '{camera}' found in {undist_dir}\n"
            f"  Expected a file matching: *-{camera}_undistort.mp4"
        )
    if len(matches) > 1:
        print(f"[io_utils] Warning: multiple matches for '{camera}' in {undist_dir}, using first: {matches[0].name}")

    return matches[0]


# ── Detection JSON ────────────────────────────────────────────────────────────

def save_detection_json(frame_dict: dict, out_path: Path):
    """
    Write a per-frame detection dict to a JSON file.

    Parameters
    ----------
    frame_dict : output of perception/export.py build_frame_dict()
    out_path   : destination file, e.g. Outputs/Detections/scene1/frame_000001.json
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(frame_dict, f, indent=2)
    


# def load_detection_json(json_path: Path) -> dict:
#     """
#     Load a single per-frame detection JSON.

#     Returns
#     -------
#     dict matching the schema in perception/export.py
#     """
#     with open(json_path, "r") as f:
#         return json.load(f)


# def list_frame_jsons(detections_dir: Path) -> List[Path]:
#     """
#     Return a sorted list of all frame JSON files in a detections directory.

#     Parameters
#     ----------
#     detections_dir : e.g. Outputs/Detections/scene1/

#     Returns
#     -------
#     list[Path] sorted by frame number
#     """
#     d = Path(detections_dir)
#     if not d.exists():
#         return []
#     return sorted(d.glob("frame_*.json"))


# # ── Image save ────────────────────────────────────────────────────────────────

# def save_frame_png(frame_bgr: "np.ndarray", out_path: Path):
#     """Save a BGR numpy array as PNG."""
#     import cv2
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(out_path), frame_bgr)
