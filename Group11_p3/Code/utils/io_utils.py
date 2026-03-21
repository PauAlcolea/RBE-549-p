# """
# utils/io_utils.py
# =================
# File I/O helpers shared by both the perception pipeline and the Blender module.

# Covers:
#   - Config loading (YAML)
#   - Video frame extraction
#   - Detection JSON read/write
#   - Frame listing utilities
# """

# from pathlib import Path
# from typing import Iterator, List, Tuple
# import json
# import os


# # ── Config ────────────────────────────────────────────────────────────────────

# def load_config(config_path: str | Path = None) -> dict:
#     """
#     Load config.yaml. Searches for it relative to the Code/ directory
#     if no explicit path is given.

#     Returns
#     -------
#     dict : parsed config
#     """
#     import yaml

#     if config_path is None:
#         # Walk up from this file until we find config.yaml
#         here = Path(__file__).resolve().parent
#         for candidate in [here, here.parent, here.parent.parent]:
#             p = candidate / "config.yaml"
#             if p.exists():
#                 config_path = p
#                 break
#         else:
#             raise FileNotFoundError("config.yaml not found. Pass an explicit path.")

#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)


# # ── Video frame extraction ────────────────────────────────────────────────────

# def get_video_frames(
#     seq_dir: Path,
#     frame_skip: int = 1,
# ) -> List[Tuple[int, "np.ndarray"]]:
#     """
#     Extract frames from the undistorted video in a sequence directory.

#     Looks for a file named 'video.mp4' or any .mp4 inside seq_dir.
#     Returns a list of (frame_index, bgr_array) tuples.

#     Parameters
#     ----------
#     seq_dir    : path to e.g. Data/Sequences/Seq1/
#     frame_skip : keep every Nth frame (1 = all frames)

#     Returns
#     -------
#     list of (frame_idx, np.ndarray) — BGR images
#     """
#     import cv2

#     # Find the video file — prefer undistorted if available
#     video_path = None
#     for candidate in ["undistorted.mp4", "video.mp4"]:
#         p = Path(seq_dir) / candidate
#         if p.exists():
#             video_path = p
#             break
#     if video_path is None:
#         mp4s = list(Path(seq_dir).glob("*.mp4"))
#         if mp4s:
#             video_path = mp4s[0]
#         else:
#             raise FileNotFoundError(f"No .mp4 found in {seq_dir}")

#     cap = cv2.VideoCapture(str(video_path))
#     frames = []
#     frame_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_idx % frame_skip == 0:
#             frames.append((frame_idx, frame))
#         frame_idx += 1

#     cap.release()
#     return frames


# def frame_generator(
#     seq_dir: Path,
#     frame_skip: int = 1,
# ) -> Iterator[Tuple[int, "np.ndarray"]]:
#     """
#     Memory-efficient generator version of get_video_frames.
#     Yields (frame_idx, bgr_array) one at a time — use for long sequences.
#     """
#     import cv2

#     video_path = _find_video(Path(seq_dir))
#     cap = cv2.VideoCapture(str(video_path))
#     frame_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_idx % frame_skip == 0:
#             yield frame_idx, frame
#         frame_idx += 1

#     cap.release()


# def _find_video(seq_dir: Path) -> Path:
#     for name in ["undistorted.mp4", "video.mp4"]:
#         p = seq_dir / name
#         if p.exists():
#             return p
#     mp4s = list(seq_dir.glob("*.mp4"))
#     if mp4s:
#         return mp4s[0]
#     raise FileNotFoundError(f"No .mp4 found in {seq_dir}")


# # ── Detection JSON ────────────────────────────────────────────────────────────

# def save_detection_json(frame_dict: dict, out_path: Path):
#     """
#     Write a per-frame detection dict to a JSON file.

#     Parameters
#     ----------
#     frame_dict : output of perception/export.py build_frame_dict()
#     out_path   : destination file, e.g. Data/outputs/detections/Seq1/frame_000001.json
#     """
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w") as f:
#         json.dump(frame_dict, f, indent=2)


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
#     detections_dir : e.g. Data/outputs/detections/Seq1/

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
