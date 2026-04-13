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

import os
import shutil
import tempfile
from pathlib import Path
import subprocess


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


def frames_to_video(frames_dir: Path, out_path: Path, fps: int = 30):
    """
    Stitch a directory of PNG frames into an MP4 using ffmpeg.

    Expects frames named frame_000001.png, frame_000002.png, etc.

    Parameters
    ----------
    frames_dir : directory containing PNG frames
    out_path   : destination .mp4 path
    fps        : output frame rate
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No rendered frames found in {frames_dir}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg executable not found on PATH")

    print(f"[render] Stitching {len(frame_paths)} frames → {out_path}")

    # Reindex to a contiguous temporary sequence so ffmpeg does not depend on
    # original frame numbering (supports non-zero starts and skipped indices).
    with tempfile.TemporaryDirectory(prefix="ffmpeg_stage_", dir=str(frames_dir)) as stage_tmp:
        stage_dir = Path(stage_tmp)
        for i, src in enumerate(frame_paths):
            dst = stage_dir / f"frame_{i:06d}.png"
            try:
                os.link(str(src), str(dst))
            except OSError:
                shutil.copy2(src, dst)

        pattern = str(stage_dir / "frame_%06d.png")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
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
            "-movflags",
            "+faststart",
            str(out_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[render] ffmpeg error:\n{result.stderr}")
            raise RuntimeError(f"ffmpeg failed for {out_path}")

    print(f"[render] Video written: {out_path}")