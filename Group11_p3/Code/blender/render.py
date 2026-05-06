"""
blender/render.py
=================
Handles single-frame rendering and final PNG → MP4 stitching.

Two distinct jobs:
  1. render_frame()     — tell Blender to render the current scene to a PNG
  2. frames_to_video()  — stitch the PNG sequence into an MP4 using OpenCV

OpenCV is used for stitching because Blender's built-in video output
re-encodes everything and is slower for batch jobs. ffmpeg must be
available on PATH (it's pre-installed on most Linux clusters).
"""

import os
import shutil
import tempfile
from pathlib import Path
import subprocess
if __name__ == "__main__":
    import cv2

def get_front_video_path(scene_num: int, base_dir: Path = Path("../Data/Sequences")) -> Path:
    """
    Infers the path to the front camera video for a given scene number.
    Searches in Data/Sequences/scene#/Undist/ for a file containing 'front' and ending with .mp4.
    Returns the Path if found, else raises FileNotFoundError.
    """
    scene_dir = base_dir / f"scene{scene_num}" / "Undist"
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    for file in scene_dir.iterdir():
        if file.is_file() and "front" in file.name and file.suffix == ".mp4":
            return file
    raise FileNotFoundError(f"No front .mp4 video found in {scene_dir}")

def render_frame(cfg: dict, out_path: Path):
    """
    Render the current Blender scene to a PNG file.

    Parameters
    ----------
    cfg      : full config dict
    out_path : destination PNG path (parent directory must exist)
    """
    import bpy
    scene = bpy.context.scene
    scene.render.filepath = str(out_path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"  # keep alpha for compositing
    bpy.ops.render.render(write_still=True)


def frames_to_video(frames_dir: Path, out_path: Path, fps: int = 6, ext: str = "png"):
    """
    Stitch a directory of image frames into an MP4 using OpenCV.
    Expects frames named like frame_000001.<ext>, frame_000002.<ext>, etc.

    Parameters
    ----------
    frames_dir : directory containing PNG frames
    out_path   : destination .mp4 path
    fps        : output frame rate
    """
    import cv2
    frame_paths = sorted(frames_dir.glob(f"frame_*.{ext}"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob(f"*.{ext}"))
    if not frame_paths:
        raise RuntimeError(f"No .{ext} frames found in {frames_dir}")

    # Read the first frame to get size
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    written = 0
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[render] WARNING: Could not read frame: {frame_path}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    writer.release()
    if written == 0:
        raise RuntimeError("No frames were written to the video")
    print(f"[render] Video written: {out_path} ({written} frames)")


# def composite_overlay(background_path: Path, overlay_path: Path, out_path: Path, alpha: float = 0.85):
#     """
#     OPTIONAL: Alpha-composite a rendered overlay PNG onto the original video frame.

#     Useful for the Tesla-style look where you see the real road underneath
#     the 3D visualization. Call this after render_frame() if desired.

#     Parameters
#     ----------
#     background_path : original video frame PNG
#     overlay_path    : rendered Blender frame (RGBA)
#     out_path        : composited output PNG
#     alpha           : opacity of the overlay layer
#     """
#     # TODO: implement if desired for final visual style
#     # import cv2, numpy as np
#     # bg   = cv2.imread(str(background_path)).astype(float)
#     # ov   = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED).astype(float)
#     # a    = (ov[:, :, 3:4] / 255.0) * alpha
#     # comp = bg * (1 - a) + ov[:, :, :3] * a
#     # cv2.imwrite(str(out_path), comp.astype("uint8"))
#     raise NotImplementedError("composite_overlay not yet implemented")


def extract_frame_from_video_by_index(video_path: Path, frame_idx: int, out_path: Path):
    """
    OPTIONAL: Extract a single frame from the original video by index.

    Useful for debugging or for compositing the rendered overlay onto the
    original video frame.

    Parameters
    ----------
    video_path : input video file
    frame_idx  : zero-based frame index to extract
    out_path   : destination PNG path
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"[render] Extracted frame {frame_idx} to {out_path}")


def render_video_side_by_side(
    video1_path: Path,
    frames_dir: Path,
    out_path: Path,
    fps: int = 30,
    ext: str = "png",
    start_frame_idx: int = 0,
    skip_frames: int = 0,
):
    """
    OPTIONAL: Create a side-by-side video comparing the original footage and rendered frames.

    Useful for debugging or for final presentation.

    Parameters
    ----------
    video1_path      : original video file
    frames_dir       : directory containing rendered PNG frames
    out_path         : destination MP4 path
    fps              : output frame rate
    start_frame_idx  : zero-based frame index to start reading from in video1
    skip_frames      : stride for sampling frames from video1_path (0 or 1 = use every frame)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if start_frame_idx < 0:
        raise ValueError("start_frame_idx must be >= 0")
    if skip_frames < 0:
        raise ValueError("skip_frames must be >= 0")

    frame_paths = sorted(frames_dir.glob(f"debug_frame_*.{ext}"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob(f"*.{ext}"))
    if not frame_paths:
        raise RuntimeError(f"No .{ext} frames found in {frames_dir}")

    # skip_frames applies only to source video stepping, not rendered frame selection.
    step = max(1, skip_frames)

    cap1 = cv2.VideoCapture(str(video1_path))
    if not cap1.isOpened():
        raise RuntimeError(f"Failed to open video: {video1_path}")

    # Seek to requested start frame in the source video.
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    if fps <= 0:
        src_fps = cap1.get(cv2.CAP_PROP_FPS)
        fps = int(round(src_fps)) if src_fps and src_fps > 0 else 30

    writer = None
    written = 0

    try:
        for render_path in frame_paths:
            ok, frame1 = cap1.read()
            if not ok:
                break

            frame2 = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
            if frame2 is None:
                # keep source video in sync even if rendered frame is unreadable
                for _ in range(step - 1):
                    if not cap1.grab():
                        break
                continue

            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            if h2 != h1:
                new_w2 = max(1, int(w2 * (h1 / h2)))
                frame2 = cv2.resize(frame2, (new_w2, h1), interpolation=cv2.INTER_AREA)

            side_by_side = cv2.hconcat([frame1, frame2])

            if writer is None:
                out_h, out_w = side_by_side.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (out_w, out_h),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {out_path}")

            writer.write(side_by_side)
            written += 1

            # Advance only source video by step-1 to implement sparse sampling.
            for _ in range(step - 1):
                if not cap1.grab():
                    break
    finally:
        cap1.release()
        if writer is not None:
            writer.release()

    if written == 0:
        raise RuntimeError("No frames were written to the side-by-side video")

    print(
        f"[render] Side-by-side video written: {out_path} ({written} frames), "
        f"start_frame_idx={start_frame_idx}, skip_frames={skip_frames}"
    )



if __name__ == "__main__":
    # Example usage: just specify the scene number
    from argparse import ArgumentParser


    parser = ArgumentParser()
    parser.add_argument("--scene", type=int, required=True)
    args = parser.parse_args()

    scene_num = args.scene
    try:
        video_path = get_front_video_path(scene_num)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    frames_dir = Path(f"../Outputs/Detections/scene6/debug")
    # out_path = Path(f"../Outputs/Videos/scene{scene_num}_front_side_by_side1.mp4")
    # render_video_side_by_side(
    #     video_path,
    #     frames_dir,
    #     out_path,
    #     fps=6,
    #     ext="png",
    #     start_frame_idx=0,
    #     skip_frames=1
    # )

    frames_to_video(frames_dir, Path(f"../Outputs/Videos/scene{scene_num}_debug.mp4"), fps=30, ext="png")




