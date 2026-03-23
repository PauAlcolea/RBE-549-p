"""
run_perception.py
=================
Entry point for the perception pipeline. Run this on the SSH cluster.

Usage
-----
  # Single sequence
  python run_perception.py --seq Seq1

  # All sequences defined in config.yaml
  python run_perception.py --all

  # Override device
  python run_perception.py --seq Seq1 --device cpu

Output
------
  Writes per-frame JSON files to:
    Data/outputs/detections/<seq>/frame_XXXXXX.json
"""

import argparse
import torch
from pathlib import Path
import os
import sys

# Make sure we can import our modules regardless of where the script is run from
# sys.path.insert(0, str(Path(__file__).parent))

from utils.io_utils import load_config, frame_generator, get_video_frames#, save_detection_json
# from perception.lanes import LaneDetector
from perception.objects import ObjectDetector
# from perception.depth import DepthEstimator
# from perception.traffic import TrafficLightDetector
# from perception.signs import SignDetector
# from perception.export import build_frame_dict


def parse_args():
    parser = argparse.ArgumentParser()

    # this group makes is to that some arguments can only be called in a mutually exclusive way
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scene", 
        type=str,
        help="Single sequence name, e.g. scene1"
    )
    group.add_argument(
        "--all", 
        action="store_true", 
        help="Run all sequences from config"
    )

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument(
        "--cam",
        type=str,
        help="Camera name: [front, back, left_repeater, right_repeater]"
    )
    group2.add_argument(
        "--allcam",
        action="store_true",
        help="Run for all four cameras"
    )

    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Override device from config (cuda|cpu)"
    )
    
    # this is for overlaying over footage
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Write debug overlay images alongside JSONs"
    )

    
    
    return parser.parse_args()


def load_models(cfg, device):
    """Instantiate all detectors once — expensive, do it outside the frame loop."""
    print("[init] Loading models...")
    models = {
        # "lanes":   LaneDetector(cfg, device),
        "objects": ObjectDetector(cfg, device),
        # "depth":   DepthEstimator(cfg, device),
        # "traffic": TrafficLightDetector(cfg, device),
        # "signs":   SignDetector(cfg, device),
    }
    print("[init] All models loaded in the process of instantializing detectors.")
    return models


def process_sequence(scene_name: str, camera: str, cfg: dict, models: dict, debug: bool = False):
    """Run every active detector on every frame of one sequence and write JSONs."""

    # scene_dir is the root of this sequence, e.g. Data/Sequences/scene1/
    scene_dir = Path(cfg["paths"]["sequences_dir"]) / scene_name

    # Output dir per sequence + camera, e.g. Outputs/Detections/scene1/front/
    out_dir = Path(cfg["paths"]["detections_dir"]) / scene_name / camera
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{scene_name}] Camera: {camera}")

    # Use the generator so we never load all frames into RAM at once
    for i, (frame_idx, frame_bgr) in enumerate(
        frame_generator(scene_dir, camera=camera, frame_skip=cfg["perception"]["frame_skip"])
    ):
        # --- Run detectors ---
        # object_results = models["objects"].detect(frame_bgr)
        # lane_results    = models["lanes"].detect(frame_bgr)
        # depth_map       = models["depth"].estimate(frame_bgr)
        # object_results  = models["depth"].lift_to_3d(object_results, depth_map, cfg)
        # traffic_results = models["traffic"].detect(frame_bgr, object_results)
        # sign_results    = models["signs"].detect(frame_bgr, object_results)

        # --- Build and save JSON ---
        # frame_dict = build_frame_dict(
        #     frame_idx=frame_idx,
        #     fps=cfg["blender"]["fps"],
        #     lanes=lane_results,
        #     objects=object_results,
        #     traffic_lights=traffic_results,
        #     stop_signs=sign_results,
        # )
        # save_detection_json(frame_dict, out_dir / f"frame_{frame_idx:06d}.json")

        if (i + 1) % 50 == 0:
            print(f"  [{scene_name}/{camera}] {i+1} frames processed")

    print(f"[{scene_name}] Done. JSONs saved to {out_dir}")
    return


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[init] Using device: {device}")

    args = parse_args()
    cfg = load_config("config.yaml")

    if args.device:
        device = args.device

    # for the sequences, get all of them unless the argument just specified one
    scenes = cfg["sequences"] if args.all else [args.scene]

    cameras = cfg["cameras"] if args.allcam else [args.cam]

    # instantiate all of the detectors
    models = load_models(cfg, device)

    # process the sequences
    for scene in scenes:
        for camera in cameras:  # this might break at the time to do for all the cameras
            process_sequence(scene, camera, cfg, models, debug=args.debug)

    print("\n[done] All sequences processed.")
    return

if __name__ == "__main__":
    main()
