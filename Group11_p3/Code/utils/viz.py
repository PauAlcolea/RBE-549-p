"""
utils/viz.py
============
OpenCV debug visualization helpers.

These are NOT part of the final Blender output — they're for sanity-checking
detector outputs during development before wiring up the Blender pipeline.

Usage example:
    from utils.viz import draw_detections, draw_lanes, show_or_save
    debug_frame = draw_detections(frame_bgr, vehicles, pedestrians)
    debug_frame = draw_lanes(debug_frame, lanes)
    show_or_save(debug_frame, "debug_output/Seq1/frame_000001.png")
"""

# from __future__ import annotations
# from typing import List
import numpy as np
import cv2
from pathlib import Path



# ── Colors (BGR) ──────────────────────────────────────────────────────────────
_COLOR_CAR   = (255, 160,  40)   # orange
_COLOR_PED   = ( 60, 230,  60)   # green
_COLOR_LANE_WHITE  = (255, 255, 255)
_COLOR_LANE_YELLOW = ( 30, 210, 255)
_COLOR_TL_RED    = ( 50,  50, 220)
_COLOR_TL_YELLOW = ( 30, 210, 255)
_COLOR_TL_GREEN  = ( 60, 220,  60)
_COLOR_SIGN  = ( 50,  50, 220)   # red



def draw_detections(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes for all detections on a copy of the frame.

    Parameters
    ----------
    frame_bgr  : original BGR image
    detections : list[Detection] — mix of vehicles and pedestrians

    Returns
    -------
    np.ndarray : annotated BGR image (original is not modified)
    """
    out = frame_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = _COLOR_PED if det.label == "person" else _COLOR_CAR

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label: show depth once DepthEstimator is wired in, otherwise just label + conf
        if det.depth_m > 0:
            text = f"{det.label} {det.depth_m:.1f}m"
        else:
            text = f"{det.label} {det.confidence:.2f}"

        if getattr(det, "direction", "unknown") != "unknown":
            text = f"{text} {det.direction}"

        # Draw a filled background behind the text so it's readable on any frame
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return out


# def draw_lanes(frame_bgr: np.ndarray, lanes: list) -> np.ndarray:
#     """
#     Draw lane polylines on a copy of the frame.

#     Parameters
#     ----------
#     frame_bgr : original BGR image
#     lanes     : list[Lane]

#     Returns
#     -------
#     np.ndarray : annotated BGR image
#     """
#     out = frame_bgr.copy()

#     for lane in lanes:
#         color = _COLOR_LANE_YELLOW if lane.color == "yellow" else _COLOR_LANE_WHITE
#         pts = np.array([[int(x), int(y)] for x, y in lane.points], dtype=np.int32)
#         thickness = 2
#         if lane.style == "dashed":
#             for i in range(0, len(pts) - 1, 2):
#                 cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)
#         else:
#             cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness)

#     return out


def draw_traffic_lights(frame_bgr: np.ndarray, traffic_lights: list) -> np.ndarray:
    """Draw traffic light bboxes with their classified color."""
    out = frame_bgr.copy()
    color_map = {
        "red":     _COLOR_TL_RED,
        "yellow":  _COLOR_TL_YELLOW,
        "green":   _COLOR_TL_GREEN,
        "unknown": (128, 128, 128),
    }
    for tl in traffic_lights:
        x1, y1, x2, y2 = [int(v) for v in tl.bbox]
        c = color_map.get(tl.color, (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        cv2.putText(out, tl.color, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return out


def draw_signs(frame_bgr: np.ndarray, signs: list) -> np.ndarray:
    """Draw sign bboxes."""
    out = frame_bgr.copy()
    for sign in signs:
        x1, y1, x2, y2 = [int(v) for v in sign.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_SIGN, 2)
        cv2.putText(out, sign.label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_SIGN, 1)
    return out


# def draw_depth_map(depth_map: np.ndarray) -> np.ndarray:
#     """
#     Normalize and colorize a depth map for visualization.
    # Returns a BGR image the same size as the input.
    # """
    # normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    # gray = normalized.astype(np.uint8)
    # return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)


def show_or_save(frame_bgr: np.ndarray, save_path: str = None):
    """
    Save an annotated frame to disk, or display it interactively.

    On the SSH cluster you will almost always want save_path — there's no
    display. Locally you can omit it to get a popup window.

    Parameters
    ----------
    frame_bgr : annotated BGR image
    save_path : if provided, write PNG here; otherwise imshow
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), frame_bgr)
    else:
        cv2.imshow("debug", frame_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
