"""
perception/export.py
====================
Assembles all detector outputs into the canonical per-frame JSON structure
that the Blender scripting module reads.

possible JSON schema (one file per frame):
{
  "frame": int,
  "timestamp": float,           // seconds
  "lanes": [
    {"points": [[x,y],...], "color": "white"|"yellow", "style": "solid"|"dashed"}
  ],
  "vehicles": [
    {"bbox": [x1,y1,x2,y2], "class": "car", "depth_m": float, "position_3d": [x,y,z]}
  ],
  "pedestrians": [
    {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
  ],
  "traffic_lights": [
        {"bbox": [x1,y1,x2,y2], "color": "red"|"yellow"|"green", "depth_m": float, "position_3d": [x,y,z]}
  ],
  "stop_signs": [
    {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
  ]
}
"""


def _serialize_lane(lane) -> dict:
    """Normalize lane objects/dicts to the JSON lane schema."""
    if isinstance(lane, dict):
        points = lane.get("points", [])
        color = lane.get("color", "white")
        confidence = lane.get("confidence", None)
    else:
        points = getattr(lane, "points", [])
        color = getattr(lane, "color", "white")
        confidence = getattr(lane, "confidence", None)

    norm_points = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x, y = pt[0], pt[1]
        norm_points.append([float(x), float(y)])

    lane_out = {
        "points": norm_points,
        "color": str(color),
    }

    if confidence is not None:
        lane_out["confidence"] = float(confidence)

    return lane_out

def build_frame_dict(
    frame_idx: int,
    fps: float,
    lanes: list,
    objects: list,
    traffic_lights: list,
    stop_signs: list,
) -> dict:
    """
    Convert all detector outputs for one frame into the shared JSON schema.

    Parameters
    ----------
    frame_idx : int
    fps : float          From config, used to compute timestamp.
    lanes : list[Lane]
    objects : list[Detection]   Includes both vehicles and pedestrians.
    traffic_lights : list[TrafficLight]
    stop_signs : list[Sign]

    Returns
    -------
    dict matching the schema above.
    """
    vehicles        = [d for d in objects if d.label in {"bicycle", "car", "motorcycle", "bus", "truck"}]
    pedestrians     = [d for d in objects if d.label == "person"]
    lanes_out = []
    for lane in lanes:
        lane_dict = _serialize_lane(lane)
        if len(lane_dict["points"]) >= 2:
            lanes_out.append(lane_dict)


    return {
        "frame":     frame_idx,
        "timestamp": round(frame_idx / fps, 4),

        "lanes": lanes_out,

        "vehicles": [
            {
                "bbox":        [round(v, 2) for v in det.bbox],
                "class":       det.label,
                "direction":   det.direction,
                "depth_m":     round(det.depth_m, 3),
                "position_3d": [round(v, 3) for v in det.position_3d],
                "heading_rad": round(det.heading_rad, 4),
            }
            for det in vehicles
        ],

        "pedestrians": [
            {
                "bbox":        [round(v, 2) for v in det.bbox],
                "depth_m":     round(det.depth_m, 3),
                "position_3d": [round(v, 3) for v in det.position_3d],
            }
            for det in pedestrians
        ],

        "traffic_lights": [
            {
                "bbox":    [round(v, 2) for v in tl.bbox],
                "color":   tl.color,
                "depth_m": round(tl.depth_m, 3),
                "position_3d": [round(v, 3) for v in tl.position_3d],
            }
            for tl in traffic_lights
        ],

        "stop_signs": [
            {
                "bbox":        [round(v, 2) for v in sign.bbox],
                "depth_m":     round(sign.depth_m, 3),
                "position_3d": [round(v, 3) for v in sign.position_3d],
            }
            for sign in stop_signs
        ],
    }
