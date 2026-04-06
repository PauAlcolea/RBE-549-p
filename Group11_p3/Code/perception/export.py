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
    {
      "bbox": [x1,y1,x2,y2],
      "class": "car",
      "depth_m": float,
      "position_3d": [x,y,z],
      "heading_rad": float  // optional, added when vehicle orientation is available
    }
  ],
  "pedestrians": [
    {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
  ],
  "traffic_lights": [
      {"bbox": [x1,y1,x2,y2], "color": "red"|"yellow"|"green", "depth_m": float, "position_3d": [x,y,z], "traffic_light_style": "standard_vertical"|"wide_green_arrow_candidate"|"square_arrow_signal_candidate"}
  ],
    "stop_signs": [
        {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
    ],
    "traffic_cones": [
        {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
    ],
    "trash_cans": [
        {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
    ],
    "traffic_poles": [
        {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
    ],
    "non_coco_objects": [
        {"class": "traffic_cone|trash_can|traffic_pole|...", "bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
  ]
}
"""

_VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "bus", "truck", "sedan", "hatchback", "suv", "pickuptruck", "pickup_truck"}


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


def _serialize_vehicle(det) -> dict:
    """normalize vehicles and add the radian heading if it has one"""
    vehicle = {
        "bbox": [round(v, 2) for v in det.bbox],
        "class": det.label,
        "depth_m": round(det.depth_m, 3),
        "position_3d": [round(v, 3) for v in det.position_3d],
    }

    heading_rad = getattr(det, "heading_rad", None)
    if heading_rad is not None:
        vehicle["heading_rad"] = round(float(heading_rad), 4)

    return vehicle


def _serialize_pedestrian(det) -> dict:
    ped = {
        "bbox": [round(v, 2) for v in det.bbox],
        "depth_m": round(det.depth_m, 3),
        "position_3d": [round(v, 3) for v in det.position_3d],
    }

    pymaf_track_id = getattr(det, "pymaf_track_id", None)
    if pymaf_track_id is not None:
        ped["pymaf_track_id"] = int(pymaf_track_id)
        ped["pymaf_match_iou"] = round(float(getattr(det, "pymaf_match_iou", 0.0)), 4)

    smpl_pose = getattr(det, "smpl_pose", None)
    if smpl_pose is not None:
        ped["smpl_pose"] = [round(float(v), 6) for v in smpl_pose]

    smpl_betas = getattr(det, "smpl_betas", None)
    if smpl_betas is not None:
        ped["smpl_betas"] = [round(float(v), 6) for v in smpl_betas]

    smpl_joints3d = getattr(det, "smpl_joints3d", None)
    if smpl_joints3d is not None:
        ped["smpl_joints3d"] = [
            [round(float(coord), 6) for coord in joint]
            for joint in smpl_joints3d
        ]

    return ped

def build_frame_dict(
    frame_idx: int,
    fps: float,
    lanes: list,
    objects: list,
    traffic_lights: list,
    stop_signs: list,
    non_coco_objects: list,
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
    vehicles        = [d for d in objects if d.label in _VEHICLE_LABELS]
    pedestrians     = [d for d in objects if d.label == "person"]
    non_coco_records = []
    bucket_map = {}

    for det in non_coco_objects:
        entry = {
            "class": det.label,
            "bbox": [round(v, 2) for v in det.bbox],
            "depth_m": round(det.depth_m, 3),
            "position_3d": [round(v, 3) for v in det.position_3d],
        }
        non_coco_records.append(entry)

        bucket = getattr(det, "export_bucket", "non_coco_objects")
        if bucket not in bucket_map:
            bucket_map[bucket] = []
        bucket_entry = {
            "bbox": entry["bbox"],
            "depth_m": entry["depth_m"],
            "position_3d": entry["position_3d"],
        }
        if bucket == "speed_limit_signs":
            speed_value = getattr(det, "speed_value", None)
            bucket_entry["speed_value"] = int(speed_value) if speed_value is not None else None
            bucket_entry["ocr_confidence"] = round(float(getattr(det, "ocr_confidence", 0.0)), 4)
            bucket_entry["ocr_raw_text"] = str(getattr(det, "ocr_raw_text", ""))

        bucket_map[bucket].append(bucket_entry)

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
            _serialize_vehicle(det)
            for det in vehicles
        ],

        "pedestrians": [
            _serialize_pedestrian(det)
            for det in pedestrians
        ],

        "traffic_lights": [
            {
                "bbox":    [round(v, 2) for v in tl.bbox],
                "color":   tl.color,
                "depth_m": round(tl.depth_m, 3),
                "position_3d": [round(v, 3) for v in tl.position_3d],
                "traffic_light_style": str(getattr(tl, "traffic_light_style", "standard_vertical")),
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

        "traffic_cones": bucket_map.get("traffic_cones", []),
        "trash_cans": bucket_map.get("trash_cans", []),
        "traffic_poles": bucket_map.get("traffic_poles", []),
        "speed_limit_signs": bucket_map.get("speed_limit_signs", []),
        "non_coco_objects": non_coco_records,
    }
