"""
perception/tracker.py
=====================
ByteTrack wrapper for assigning stable track IDs to frame-level detections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import sys

import numpy as np


@dataclass
class _TrackableDetection:
    det: object
    class_id: int


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


class ByteTrackWrapper:
    """Apply ByteTrack IDs to detection objects in-place."""

    def __init__(self, cfg: dict):
        tracker_cfg = cfg.get("perception", {}).get("tracker", {})
        self.enabled = bool(tracker_cfg.get("enabled", False))
        self.assign_iou = float(tracker_cfg.get("assignment_iou", 0.1))
        self.track_objects = bool(tracker_cfg.get("track_objects", True))
        self.track_non_coco = bool(tracker_cfg.get("track_non_coco", True))

        include_labels = tracker_cfg.get("include_labels", [])
        exclude_labels = tracker_cfg.get("exclude_labels", [])
        self.include_labels = {
            str(label).strip()
            for label in include_labels
            if str(label).strip()
        }
        self.exclude_labels = {
            str(label).strip()
            for label in exclude_labels
            if str(label).strip()
        }

        self._label_to_class_id: Dict[str, int] = {}
        self._next_class_id = 0
        self._tracker = None

        if not self.enabled:
            return

        dart_dir = Path(__file__).resolve().parent / "DART"
        if str(dart_dir) not in sys.path:
            sys.path.insert(0, str(dart_dir))

        from sam3.tracking import BYTETracker

        self._tracker = BYTETracker(
            track_thresh=float(tracker_cfg.get("track_thresh", 0.5)),
            match_thresh=float(tracker_cfg.get("match_thresh", 0.5)),
            second_match_thresh=float(tracker_cfg.get("second_match_thresh", 0.4)),
            lost_match_thresh=float(tracker_cfg.get("lost_match_thresh", 0.3)),
            max_time_lost=int(tracker_cfg.get("max_time_lost", 30)),
            min_hits=int(tracker_cfg.get("min_hits", 3)),
            duplicate_iou_thresh=float(tracker_cfg.get("duplicate_iou_thresh", 0.85)),
            class_agnostic_nms_thresh=float(tracker_cfg.get("class_agnostic_nms_thresh", 1.0)),
        )

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()

    def _is_trackable(self, det: object) -> bool:
        label = str(getattr(det, "label", "")).strip()
        if not label:
            return False
        if label in self.exclude_labels:
            return False
        if self.include_labels and label not in self.include_labels:
            return False
        bbox = getattr(det, "bbox", None)
        if bbox is None or len(bbox) != 4:
            return False
        return True

    def _class_id(self, label: str) -> int:
        if label not in self._label_to_class_id:
            self._label_to_class_id[label] = self._next_class_id
            self._next_class_id += 1
        return self._label_to_class_id[label]

    def _collect(self, object_dets: list, non_coco_dets: list) -> List[_TrackableDetection]:
        combined: List[_TrackableDetection] = []

        if self.track_objects:
            for det in object_dets:
                setattr(det, "track_id", None)
                if not self._is_trackable(det):
                    continue
                label = str(getattr(det, "label", "")).strip()
                combined.append(_TrackableDetection(det=det, class_id=self._class_id(label)))

        if self.track_non_coco:
            for det in non_coco_dets:
                setattr(det, "track_id", None)
                if not self._is_trackable(det):
                    continue
                label = str(getattr(det, "label", "")).strip()
                combined.append(_TrackableDetection(det=det, class_id=self._class_id(label)))

        return combined

    def update(self, object_dets: list, non_coco_dets: list) -> int:
        """Track detections for one frame and assign det.track_id in-place."""
        if self._tracker is None:
            return 0

        trackable = self._collect(object_dets, non_coco_dets)
        if not trackable:
            self._tracker.update(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
            return 0

        boxes = np.asarray(
            [np.asarray(getattr(item.det, "bbox"), dtype=np.float32) for item in trackable],
            dtype=np.float32,
        )
        scores = np.asarray(
            [float(getattr(item.det, "confidence", 0.0)) for item in trackable],
            dtype=np.float32,
        )
        class_ids = np.asarray([item.class_id for item in trackable], dtype=np.int64)

        tracks = self._tracker.update(boxes, scores, class_ids)

        available = set(range(len(trackable)))
        for track in tracks:
            track_box = getattr(track, "box_xyxy", None)
            track_class = int(getattr(track, "class_id", -1))
            if track_box is None:
                continue

            best_idx = None
            best_iou = 0.0
            for idx in list(available):
                cand = trackable[idx]
                if cand.class_id != track_class:
                    continue
                iou = _bbox_iou(getattr(cand.det, "bbox"), track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is None or best_iou < self.assign_iou:
                continue

            setattr(trackable[best_idx].det, "track_id", int(getattr(track, "track_id", -1)))
            available.remove(best_idx)

        return len(tracks)
