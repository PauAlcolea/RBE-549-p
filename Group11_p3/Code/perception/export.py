# """
# perception/export.py
# ====================
# Assembles all detector outputs into the canonical per-frame JSON structure
# that the Blender scripting module reads.

# possible JSON schema (one file per frame):
# {
#   "frame": int,
#   "timestamp": float,           // seconds
#   "lanes": [
#     {"points": [[x,y],...], "color": "white"|"yellow", "style": "solid"|"dashed"}
#   ],
#   "vehicles": [
#     {"bbox": [x1,y1,x2,y2], "class": "car", "depth_m": float, "position_3d": [x,y,z]}
#   ],
#   "pedestrians": [
#     {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
#   ],
#   "traffic_lights": [
#     {"bbox": [x1,y1,x2,y2], "color": "red"|"yellow"|"green", "depth_m": float}
#   ],
#   "stop_signs": [
#     {"bbox": [x1,y1,x2,y2], "depth_m": float, "position_3d": [x,y,z]}
#   ]
# }
# """

# def build_frame_dict(
#     frame_idx: int,
#     fps: float,
#     lanes: list,
#     objects: list,
#     traffic_lights: list,
#     stop_signs: list,
# ) -> dict:
#     """
#     Convert all detector outputs for one frame into the shared JSON schema.

#     Parameters
#     ----------
#     frame_idx : int
#     fps : float          From config, used to compute timestamp.
#     lanes : list[Lane]
#     objects : list[Detection]   Includes both vehicles and pedestrians.
#     traffic_lights : list[TrafficLight]
#     stop_signs : list[Sign]

#     Returns
#     -------
#     dict matching the schema above.
#     """
#     vehicles    = [d for d in objects if d.label != "person"]
#     pedestrians = [d for d in objects if d.label == "person"]

#     return {
#         "frame":     frame_idx,
#         "timestamp": round(frame_idx / fps, 4),

#         "lanes": [
#             {
#                 "points": [[float(x), float(y)] for x, y in lane.points],
#                 "color":  lane.color,
#                 "style":  lane.style,
#             }
#             for lane in lanes
#         ],

#         "vehicles": [
#             {
#                 "bbox":        [round(v, 2) for v in det.bbox],
#                 "class":       det.label,
#                 "depth_m":     round(det.depth_m, 3),
#                 "position_3d": [round(v, 3) for v in det.position_3d],
#             }
#             for det in vehicles
#         ],

#         "pedestrians": [
#             {
#                 "bbox":        [round(v, 2) for v in det.bbox],
#                 "depth_m":     round(det.depth_m, 3),
#                 "position_3d": [round(v, 3) for v in det.position_3d],
#             }
#             for det in pedestrians
#         ],

#         "traffic_lights": [
#             {
#                 "bbox":    [round(v, 2) for v in tl.bbox],
#                 "color":   tl.color,
#                 "depth_m": round(tl.depth_m, 3),
#             }
#             for tl in traffic_lights
#         ],

#         "stop_signs": [
#             {
#                 "bbox":        [round(v, 2) for v in sign.bbox],
#                 "depth_m":     round(sign.depth_m, 3),
#                 "position_3d": [round(v, 3) for v in sign.position_3d],
#             }
#             for sign in stop_signs
#         ],
#     }
