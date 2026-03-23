# """
# utils/viz.py
# ============
# OpenCV debug visualization helpers.

# These are NOT part of the final Blender output — they're for sanity-checking
# detector outputs during development before wiring up the Blender pipeline.

# Usage example:
#     from utils.viz import draw_detections, draw_lanes, show_or_save
#     debug_frame = draw_detections(frame_bgr, vehicles, pedestrians)
#     debug_frame = draw_lanes(debug_frame, lanes)
#     show_or_save(debug_frame, "debug_output/Seq1/frame_000001.png")
# """

# from __future__ import annotations
# from typing import List
# import numpy as np


# # ── Colors (BGR) ──────────────────────────────────────────────────────────────
# _COLOR_CAR   = (255, 160,  40)   # orange
# _COLOR_PED   = ( 60, 230,  60)   # green
# _COLOR_LANE_WHITE  = (255, 255, 255)
# _COLOR_LANE_YELLOW = ( 30, 210, 255)
# _COLOR_TL_RED    = ( 50,  50, 220)
# _COLOR_TL_YELLOW = ( 30, 210, 255)
# _COLOR_TL_GREEN  = ( 60, 220,  60)
# _COLOR_SIGN  = ( 50,  50, 220)   # red


# def draw_detections(
#     frame_bgr: np.ndarray,
#     vehicles: list,
#     pedestrians: list,
# ) -> np.ndarray:
#     """
#     Draw bounding boxes for vehicles and pedestrians on a copy of the frame.

#     Parameters
#     ----------
#     frame_bgr    : original BGR image
#     vehicles     : list[Detection] with label="car"
#     pedestrians  : list[Detection] with label="person"

#     Returns
#     -------
#     np.ndarray : annotated BGR image
#     """
#     # import cv2
#     # out = frame_bgr.copy()

#     # for det in vehicles:
#     #     x1, y1, x2, y2 = [int(v) for v in det.bbox]
#     #     cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_CAR, 2)
#     #     label = f"car {det.depth_m:.1f}m" if det.depth_m > 0 else "car"
#     #     cv2.putText(out, label, (x1, y1 - 6),
#     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_CAR, 1)

#     # for det in pedestrians:
#     #     x1, y1, x2, y2 = [int(v) for v in det.bbox]
#     #     cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_PED, 2)
#     #     label = f"ped {det.depth_m:.1f}m" if det.depth_m > 0 else "ped"
#     #     cv2.putText(out, label, (x1, y1 - 6),
#     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_PED, 1)

#     # return out
#     return


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
#     # import cv2
#     # out = frame_bgr.copy()

#     # for lane in lanes:
#     #     color = _COLOR_LANE_YELLOW if lane.color == "yellow" else _COLOR_LANE_WHITE
#     #     pts = np.array([[int(x), int(y)] for x, y in lane.points], dtype=np.int32)
#     #     thickness = 2
#     #     if lane.style == "dashed":
#     #         # Draw dashed line by alternating draw/skip segments
#     #         for i in range(0, len(pts) - 1, 2):
#     #             cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)
#     #     else:
#     #         cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness)

#     # return out
#     return


# def draw_traffic_lights(frame_bgr: np.ndarray, traffic_lights: list) -> np.ndarray:
#     # """Draw traffic light bboxes with their classified color."""
#     # import cv2
#     # out = frame_bgr.copy()
#     # color_map = {
#     #     "red":     _COLOR_TL_RED,
#     #     "yellow":  _COLOR_TL_YELLOW,
#     #     "green":   _COLOR_TL_GREEN,
#     #     "unknown": (128, 128, 128),
#     # }
#     # for tl in traffic_lights:
#     #     x1, y1, x2, y2 = [int(v) for v in tl.bbox]
#     #     c = color_map.get(tl.color, (128, 128, 128))
#     #     cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
#     #     cv2.putText(out, tl.color, (x1, y1 - 6),
#     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
#     # return out
#     return


# def draw_signs(frame_bgr: np.ndarray, signs: list) -> np.ndarray:
#     """Draw sign bboxes."""
#     # import cv2
#     # out = frame_bgr.copy()
#     # for sign in signs:
#     #     x1, y1, x2, y2 = [int(v) for v in sign.bbox]
#     #     cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_SIGN, 2)
#     #     cv2.putText(out, sign.label, (x1, y1 - 6),
#     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_SIGN, 1)
#     # return out
#     return


# def draw_depth_map(depth_map: np.ndarray) -> np.ndarray:
#     """
#     Normalize and colorize a depth map for visualization.

#     Returns a BGR image.
#     """
#     # import cv2
#     # normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
#     # gray = normalized.astype(np.uint8)
#     # return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
#     return


# def draw_all(
#     frame_bgr: np.ndarray,
#     frame_data: dict,
# ) -> np.ndarray:
#     """
#     Convenience wrapper: draw everything from a frame JSON dict onto the frame.

#     Parameters
#     ----------
#     frame_bgr  : raw video frame
#     frame_data : loaded JSON dict (from io_utils.load_detection_json)

#     Returns
#     -------
#     Fully annotated debug frame.
#     """
#     # # Import dataclasses locally to reconstruct objects from dicts
#     # from perception.objects import Detection
#     # from perception.lanes import Lane
#     # from perception.traffic import TrafficLight
#     # from perception.signs import Sign

#     # vehicles = [
#     #     Detection(label=v["class"], bbox=v["bbox"], confidence=1.0,
#     #               depth_m=v.get("depth_m", 0.0))
#     #     for v in frame_data.get("vehicles", [])
#     # ]
#     # peds = [
#     #     Detection(label="person", bbox=p["bbox"], confidence=1.0,
#     #               depth_m=p.get("depth_m", 0.0))
#     #     for p in frame_data.get("pedestrians", [])
#     # ]
#     # lanes = [
#     #     Lane(points=[tuple(pt) for pt in l["points"]],
#     #          color=l["color"], style=l["style"])
#     #     for l in frame_data.get("lanes", [])
#     # ]
#     # lights = [
#     #     TrafficLight(bbox=t["bbox"], color=t["color"], confidence=1.0,
#     #                  depth_m=t.get("depth_m", 0.0))
#     #     for t in frame_data.get("traffic_lights", [])
#     # ]
#     # signs = [
#     #     Sign(label=s.get("label", "stop_sign"), bbox=s["bbox"],
#     #          confidence=1.0, depth_m=s.get("depth_m", 0.0))
#     #     for s in frame_data.get("stop_signs", [])
#     # ]

#     # out = frame_bgr.copy()
#     # out = draw_lanes(out, lanes)
#     # out = draw_detections(out, vehicles, peds)
#     # out = draw_traffic_lights(out, lights)
#     # out = draw_signs(out, signs)
#     # return out
#     return


# def show_or_save(frame_bgr: np.ndarray, save_path: str = None):
#     """
#     Display a frame with cv2.imshow (if running interactively) or save to disk.

#     Parameters
#     ----------
#     frame_bgr : annotated BGR image
#     save_path : if provided, write PNG to this path instead of showing
#     """
#     # import cv2
#     # from pathlib import Path

#     # if save_path:
#     #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#     #     cv2.imwrite(save_path, frame_bgr)
#     # else:
#     #     cv2.imshow("debug", frame_bgr)
#     #     cv2.waitKey(0)
#     #     cv2.destroyAllWindows()
#     return
