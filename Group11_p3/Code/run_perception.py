# """
# run_perception.py
# =================
# Entry point for the perception pipeline. Run this on the SSH cluster.

# Usage
# -----
#   # Single sequence
#   python run_perception.py --seq Seq1

#   # All sequences defined in config.yaml
#   python run_perception.py --all

#   # Override device
#   python run_perception.py --seq Seq1 --device cpu

# Output
# ------
#   Writes per-frame JSON files to:
#     Data/outputs/detections/<seq>/frame_XXXXXX.json
# """

# import argparse
# import os
# import sys
# from pathlib import Path

# # Make sure we can import our modules regardless of where the script is run from
# sys.path.insert(0, str(Path(__file__).parent))

# from utils.io_utils import load_config, get_video_frames, save_detection_json
# from perception.lanes import LaneDetector
# from perception.objects import ObjectDetector
# from perception.depth import DepthEstimator
# from perception.traffic import TrafficLightDetector
# from perception.signs import SignDetector
# from perception.export import build_frame_dict


# def parse_args():
#     # parser = argparse.ArgumentParser(description="EinsteinVision perception pipeline")
#     # group = parser.add_mutually_exclusive_group(required=True)
#     # group.add_argument("--seq", type=str, help="Single sequence name, e.g. Seq1")
#     # group.add_argument("--all", action="store_true", help="Run all sequences from config")
#     # parser.add_argument("--device", type=str, default=None,
#     #                     help="Override device from config (cuda|cpu)")
#     # parser.add_argument("--debug", action="store_true",
#     #                     help="Write debug overlay images alongside JSONs")
#     # return parser.parse_args()
#     return


# def load_models(cfg, device):
#     # """Instantiate all detectors once — expensive, do it outside the frame loop."""
#     # print("[init] Loading models...")
#     # models = {
#     #     "lanes":   LaneDetector(cfg, device),
#     #     "objects": ObjectDetector(cfg, device),
#     #     "depth":   DepthEstimator(cfg, device),
#     #     "traffic": TrafficLightDetector(cfg, device),
#     #     "signs":   SignDetector(cfg, device),
#     # }
#     # print("[init] All models loaded.")
#     # return models
#     return


# def process_sequence(seq_name: str, cfg: dict, models: dict, debug: bool = False):
#     """Run every detector on every frame of one sequence and write JSONs."""
#     # seq_video_path = Path(cfg["paths"]["sequences_dir"]) / seq_name
#     # out_dir = Path(cfg["paths"]["detections_dir"]) / seq_name
#     # out_dir.mkdir(parents=True, exist_ok=True)

#     # ### TODO: point this at the undistorted video file once data is downloaded
#     # frames = get_video_frames(seq_video_path, frame_skip=cfg["perception"]["frame_skip"])
#     # print(f"[{seq_name}] Processing {len(frames)} frames...")

#     # for i, (frame_idx, frame_bgr) in enumerate(frames):
#     #     # --- Run detectors ---
#     #     lane_results    = models["lanes"].detect(frame_bgr)
#     #     object_results  = models["objects"].detect(frame_bgr)
#     #     depth_map       = models["depth"].estimate(frame_bgr)

#     #     # Enrich object results with depth and 3D position
#     #     object_results  = models["depth"].lift_to_3d(object_results, depth_map, cfg)

#     #     traffic_results = models["traffic"].detect(frame_bgr, object_results)
#     #     sign_results    = models["signs"].detect(frame_bgr, object_results)

#     #     # --- Build and save JSON ---
#     #     frame_dict = build_frame_dict(
#     #         frame_idx=frame_idx,
#     #         fps=cfg["blender"]["fps"],
#     #         lanes=lane_results,
#     #         objects=object_results,
#     #         traffic_lights=traffic_results,
#     #         stop_signs=sign_results,
#     #     )
#     #     save_detection_json(frame_dict, out_dir / f"frame_{frame_idx:06d}.json")

#     #     if (i + 1) % 50 == 0:
#     #         print(f"  [{seq_name}] {i+1}/{len(frames)} frames done")

#     # print(f"[{seq_name}] Done. JSONs saved to {out_dir}")
#     return


# def main():
#     # args = parse_args()
#     # cfg = load_config()

#     # device = args.device or cfg["perception"]["device"]
#     # print(f"[init] Using device: {device}")

#     # sequences = cfg["sequences"] if args.all else [args.seq]
#     # models = load_models(cfg, device)

#     # for seq in sequences:
#     #     process_sequence(seq, cfg, models, debug=args.debug)

#     # print("\n[done] All sequences processed.")
#     return


# if __name__ == "__main__":
#     main()
