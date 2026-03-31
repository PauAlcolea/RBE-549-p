"""
blender/render.py
=================
Handles single-frame rendering and final PNG → MP4 stitching.

Two distinct jobs:
  1. render_frame()     — tell Blender to render the current scene to a PNG
  2. frames_to_video()  — stitch the PNG sequence into an MP4 using ffmpeg

ffmpeg is used for stitching because Blender's built-in video output
re-encodes everything and is slower for batch jobs. ffmpeg must be
available on PATH (it's pre-installed on most Linux clusters).
"""

from pathlib import Path
import subprocess
# import cv2


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
    Stitch a directory of image frames into an MP4 using ffmpeg.

    Expects frames named like frame_000001.<ext>, frame_000002.<ext>, etc.

    Parameters
    ----------
    frames_dir : directory containing PNG frames
    out_path   : destination .mp4 path
    fps        : output frame rate
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use glob pattern so that non-contiguous frame indices (e.g. 000000, 000030, ...)
    # are still picked up by ffmpeg in alphanumeric order.
    # This avoids the default image2 behavior of stopping at the first missing index.
    pattern = str(frames_dir / f"frame_*.{ext}")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",  # quality: 0=lossless, 23=default, 51=worst
        "-preset",
        "fast",
        str(out_path),
    ]

    print(f"[render] Stitching frames {frames_dir} → {out_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[render] ffmpeg error:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg failed for {out_path}")

    print(f"[render] Video written: {out_path}")


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


if __name__ == "__main__":
    # Example: Blender-rendered PNG frames
    frames_to_video(Path("../Outputs/Frames/scene2/front"), Path("../Outputs/Videos/scene2_front.mp4"), fps=15, ext="png")

    # Example: detection debug JPG frames
    # frames_to_video(
    #     Path("../Outputs/Detections/scene4/front/debug/"),
    #     Path("../Outputs/Videos/scene4_front_det.mp4"),
    #     fps=6,
    #     ext="jpg",
    # )



