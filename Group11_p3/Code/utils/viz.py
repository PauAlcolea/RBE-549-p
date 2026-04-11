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
from typing import Optional



# ── Colors (BGR) ──────────────────────────────────────────────────────────────
_COLOR_CAR   = (255, 160,  40)   # orange
_COLOR_PED   = ( 60, 230,  60)   # green
_COLOR_LANE_WHITE  = (255, 255, 255)
_COLOR_LANE_YELLOW = ( 30, 210, 255)
_COLOR_TL_RED    = ( 50,  50, 220)
_COLOR_TL_YELLOW = ( 30, 210, 255)
_COLOR_TL_GREEN  = ( 60, 220,  60)
_COLOR_SIGN  = ( 50,  50, 220)   # red
_COLOR_3D_BOX = (60, 220, 60)
_COLOR_3D_FRONT = (255, 120, 40)
_COLOR_CONE  = (  0, 140, 255)   # orange
_COLOR_NON_COCO = (180, 120, 255)
_COLOR_PYMAF = (255, 255, 0)     # cyan-yellow for pymaf overlays
_COLOR_MOVING = (40, 60, 230)     # red-ish for moving vehicles
_COLOR_PARKED = (60, 220, 60)     # green for parked vehicles


def _rotation_matrix_y(yaw_rad: float) -> np.ndarray:
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _create_3d_box_corners(dimensions_3d, center_3d, yaw_rad: float) -> np.ndarray:
    """
    Build the 8 corners of the oriented 3D bounding box in camera coordinates.

    Dimensions follow the forked repo convention: [height, width, length].
    """
    h, w, l = [float(v) for v in dimensions_3d]
    dx = l / 2.0
    dy = h / 2.0
    dz = w / 2.0

    corners = np.array(
        [
            [ dx,  dy,  dz],
            [ dx,  dy, -dz],
            [ dx, -dy,  dz],
            [ dx, -dy, -dz],
            [-dx,  dy,  dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [-dx, -dy, -dz],
        ],
        dtype=np.float32,
    )

    rotated = corners @ _rotation_matrix_y(yaw_rad).T
    return rotated + np.asarray(center_3d, dtype=np.float32)


def _project_points(points_3d: np.ndarray, proj_matrix: np.ndarray) -> Optional[np.ndarray]:
    points_h = np.concatenate(
        [points_3d.astype(np.float32), np.ones((points_3d.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    projected = points_h @ proj_matrix.T
    z = projected[:, 2]

    if np.any(z <= 1e-3):
        return None

    uv = projected[:, :2] / z[:, None]
    if not np.all(np.isfinite(uv)):
        return None

    return np.round(uv).astype(np.int32)


def _draw_projected_3d_box(frame_bgr: np.ndarray, det, proj_matrix: np.ndarray) -> None:
    dimensions_3d = getattr(det, "dimensions_3d", None)
    center_3d = getattr(det, "bbox_3d_location", None)
    heading_rad = getattr(det, "heading_rad", None)

    if dimensions_3d is None or center_3d is None or heading_rad is None:
        return

    try:
        corners_3d = _create_3d_box_corners(dimensions_3d, center_3d, float(heading_rad))
        box_2d = _project_points(corners_3d, proj_matrix)
    except Exception:
        return

    if box_2d is None:
        return

    edges = [
        (0, 2), (4, 6), (0, 4), (2, 6),
        (1, 3), (1, 5), (7, 3), (7, 5),
        (0, 1), (2, 3), (4, 5), (6, 7),
    ]
    for start_idx, end_idx in edges:
        p1 = tuple(box_2d[start_idx])
        p2 = tuple(box_2d[end_idx])
        cv2.line(frame_bgr, p1, p2, _COLOR_3D_BOX, 1, lineType=cv2.LINE_AA)

    front_mark = [tuple(box_2d[i]) for i in range(4)]
    cv2.line(frame_bgr, front_mark[0], front_mark[3], _COLOR_3D_FRONT, 1, lineType=cv2.LINE_AA)
    cv2.line(frame_bgr, front_mark[1], front_mark[2], _COLOR_3D_FRONT, 1, lineType=cv2.LINE_AA)


def _draw_heading_arrow(frame_bgr: np.ndarray, det, proj_matrix: np.ndarray) -> None:
    center_3d = getattr(det, "bbox_3d_location", None)
    heading_rad = getattr(det, "heading_rad", None)
    dimensions_3d = getattr(det, "dimensions_3d", None)

    if center_3d is None or heading_rad is None or dimensions_3d is None:
        return

    try:
        corners_3d = _create_3d_box_corners(dimensions_3d, center_3d, float(heading_rad))
        start = np.asarray(center_3d, dtype=np.float32)

        # Use the same "front" face that the 3D box renderer highlights so the
        # arrow and wireframe always agree on which way the vehicle faces.
        front_face_center = corners_3d[:4].mean(axis=0)
        direction = front_face_center - start
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return

        extension = max(float(dimensions_3d[2]) * 0.35, 0.5)
        end = front_face_center + (direction / norm) * extension
        projected = _project_points(np.stack([start, end], axis=0), proj_matrix)
    except Exception:
        return

    if projected is None:
        return

    cv2.arrowedLine(
        frame_bgr,
        tuple(projected[0]),
        tuple(projected[1]),
        _COLOR_3D_FRONT,
        2,
        line_type=cv2.LINE_AA,
        tipLength=0.25,
    )


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list,
    proj_matrix: np.ndarray = None,
) -> np.ndarray:
    """
    Draw bounding boxes for all detections on a copy of the frame.

    Parameters
    ----------
    frame_bgr  : original BGR image
    detections : list[Detection] — mix of vehicles and pedestrians
    proj_matrix: optional 3x4 camera projection matrix. If provided and a
                 detection has orientation metadata, a projected 3D box is drawn.

    Returns
    -------
    np.ndarray : annotated BGR image (original is not modified)
    """
    out = frame_bgr.copy()

    for det in detections:
        if proj_matrix is not None and det.label != "person":
            _draw_projected_3d_box(out, det, proj_matrix)
            _draw_heading_arrow(out, det, proj_matrix)

        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = _COLOR_PED if det.label == "person" else _COLOR_CAR
        if det.label != "person":
            if bool(getattr(det, "is_moving", False)):
                color = _COLOR_MOVING
            elif bool(getattr(det, "is_parked", False)):
                color = _COLOR_PARKED

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label: show depth once DepthEstimator is wired in, otherwise just label + conf
        track_id = getattr(det, "track_id", None)
        track_prefix = f"#{int(track_id)} " if track_id is not None else ""
        if det.depth_m > 0:
            text = f"{track_prefix}{det.label} {det.depth_m:.1f}m"
        else:
            text = f"{track_prefix}{det.label} {det.confidence:.2f}"

        heading_rad = getattr(det, "heading_rad", None)
        if heading_rad is not None and det.label != "person":
            text = f"{text} yaw={np.degrees(float(heading_rad)):.0f}deg"

        if det.label != "person":
            motion_conf = getattr(det, "motion_confidence", None)
            if motion_conf is not None:
                if bool(getattr(det, "is_moving", False)):
                    motion_state = "moving"
                elif bool(getattr(det, "is_parked", False)):
                    motion_state = "parked"
                else:
                    motion_state = "unknown"
                motion_source = str(getattr(det, "motion_source", "n/a"))
                text = f"{text} motion={motion_state} c={float(motion_conf):.2f}"

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


def draw_non_coco_objects(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Draw DART non-COCO object detections."""
    out = frame_bgr.copy()
    for det in detections:
        if det.label == "ground_text" and not bool(getattr(det, "has_only_letters", False)):
            # For debug overlays, only show ground text that matches ONLY-letter rule.
            continue

        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = _COLOR_CONE if det.label == "traffic_cone" else _COLOR_NON_COCO
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        track_id = getattr(det, "track_id", None)
        track_prefix = f"#{int(track_id)} " if track_id is not None else ""
        label = f"{track_prefix}{det.label} {det.confidence:.2f}"
        if det.label == "speed_limit_sign":
            speed_value = getattr(det, "speed_value", None)
            ocr_conf = float(getattr(det, "ocr_confidence", 0.0))
            speed_text = "null" if speed_value is None else str(speed_value)
            label = f"{det.label} v={speed_text} ocr={ocr_conf:.2f}"
        elif det.label == "ground_text":
            ocr_conf = float(getattr(det, "ocr_confidence", 0.0))
            only_hit = bool(getattr(det, "has_only_letters", False))
            hits = str(getattr(det, "only_letter_hits", ""))
            label = f"{det.label} only={int(only_hit)} hits={hits or '-'} ocr={ocr_conf:.2f}"
        cv2.putText(out, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def draw_pymaf_matches(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """
    Overlay PyMAF match diagnostics on top of person detections.

    A person is considered matched when `pymaf_track_id` is present.
    """
    out = frame_bgr.copy()
    matched = 0
    persons = 0

    for det in detections:
        if getattr(det, "label", "") != "person":
            continue
        persons += 1

        track_id = getattr(det, "pymaf_track_id", None)
        if track_id is None:
            continue
        matched += 1

        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        iou = float(getattr(det, "pymaf_match_iou", 0.0))
        has_pose = getattr(det, "smpl_pose", None) is not None

        cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_PYMAF, 2)
        text = f"pymaf id={int(track_id)} iou={iou:.2f} pose={'Y' if has_pose else 'N'}"
        cv2.putText(
            out,
            text,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            _COLOR_PYMAF,
            1,
            lineType=cv2.LINE_AA,
        )

    summary = f"PyMAF matched persons: {matched}/{persons}"
    cv2.rectangle(out, (8, 8), (330, 34), (10, 10, 10), -1)
    cv2.putText(
        out,
        summary,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _COLOR_PYMAF,
        1,
        lineType=cv2.LINE_AA,
    )
    return out


def draw_cones(frame_bgr: np.ndarray, cones: list) -> np.ndarray:
    """Backward-compatible wrapper around draw_non_coco_objects."""
    return draw_non_coco_objects(frame_bgr, cones)


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
