import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from queue import Empty
from queue import Queue as ThreadQueue

import cv2
import numpy as np


class Viewer(object):
    """OpenCV-only viewer for image and top-down trajectory visualization."""

    def __init__(self, start_process=True):
        queue_cls = MPQueue if start_process else ThreadQueue
        self.image_queue = queue_cls()
        self.pose_queue = queue_cls()

        self.pose_updates = 0
        self.image_updates = 0

        self.view_thread = None
        self.gt_points_xyz = None
        self.est_origin_xyz = None
        if start_process:
            self.start()

    def start(self):
        if self.view_thread is None:
            self.view_thread = Process(target=self.view)
            self.view_thread.start()

    def update_pose(self, pose):
        if pose is None:
            return
        self.pose_queue.put(np.asarray(pose.matrix(), dtype=np.float64))

    def update_image(self, image):
        if image is None:
            return
        if image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        self.image_queue.put(image)

    def set_groundtruth(self, points_xyz):
        if points_xyz is None:
            self.gt_points_xyz = None
            self.est_origin_xyz = None
            return
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError("Ground-truth points must have shape (N, 3)")
        self.gt_points_xyz = points_xyz
        self.est_origin_xyz = None

    def _drain_queues(self, camera, trajectory, image):
        pose = None
        try:
            while True:
                pose = self.pose_queue.get_nowait()
        except Empty:
            pass

        if pose is not None:
            trajectory.append(np.asarray(pose[:3, 3]).reshape(3,))
            camera = pose
            self.pose_updates += 1

        img = None
        try:
            while True:
                img = self.image_queue.get_nowait()
        except Empty:
            pass

        if img is not None:
            image = img.copy()
            self.image_updates += 1

        return camera, image

    def _draw_trajectory(self, traj_vis, points_xyz, gt_points_xyz=None):
        if len(points_xyz) == 0 and (gt_points_xyz is None or len(gt_points_xyz) == 0):
            return

        est_points_xyz = points_xyz
        if gt_points_xyz is not None and len(gt_points_xyz) > 0 and len(points_xyz) > 0:
            if self.est_origin_xyz is None:
                self.est_origin_xyz = np.asarray(points_xyz[0], dtype=np.float64).copy()
            # Visual-only translation so both tracks start at the same position.
            est_offset = gt_points_xyz[0] - self.est_origin_xyz
            est_points_xyz = points_xyz + est_offset

        xz_sets = []
        if gt_points_xyz is not None and len(gt_points_xyz) > 0:
            xz_sets.append(gt_points_xyz[:, [0, 2]])
        if len(est_points_xyz) > 0:
            xz_sets.append(est_points_xyz[:, [0, 2]])
        xz_all = np.vstack(xz_sets)

        # Keep a square world window so scale is stable and centered.
        # When GT is available, anchor the view on GT so estimate outliers do not shrink it.
        if gt_points_xyz is not None and len(gt_points_xyz) > 0:
            gt_xz = gt_points_xyz[:, [0, 2]]
            center = gt_xz.mean(axis=0)
            radius = np.max(np.abs(gt_xz - center))
            radius *= 1.05
        else:
            center = xz_all.mean(axis=0)
            radius = np.max(np.abs(xz_all - center))
        radius = max(radius, 0.5)

        minv = center - radius
        maxv = center + radius
        span = np.maximum(maxv - minv, 1e-6)

        if gt_points_xyz is not None and len(gt_points_xyz) > 0:
            gt_xz = gt_points_xyz[:, [0, 2]]
            gt_norm = (gt_xz - minv) / span
            gt_pix = np.zeros((len(gt_norm), 2), dtype=np.int32)
            gt_pix[:, 0] = (gt_norm[:, 0] * 560 + 20).astype(np.int32)
            gt_pix[:, 1] = (gt_norm[:, 1] * 560 + 20).astype(np.int32)
            gt_pix[:, 1] = 599 - gt_pix[:, 1]
            if len(gt_pix) > 1:
                cv2.polylines(traj_vis, [gt_pix.reshape(-1, 1, 2)], False, (220, 120, 0), 2)

        if len(est_points_xyz) > 0:
            xz = est_points_xyz[:, [0, 2]]
            norm = (xz - minv) / span
            pix = np.zeros((len(norm), 2), dtype=np.int32)
            pix[:, 0] = (norm[:, 0] * 560 + 20).astype(np.int32)
            pix[:, 1] = (norm[:, 1] * 560 + 20).astype(np.int32)
            pix[:, 1] = 599 - pix[:, 1]

            for p in pix:
                cv2.circle(traj_vis, tuple(p), 2, (0, 0, 255), -1)
            if len(pix) > 1:
                cv2.polylines(traj_vis, [pix.reshape(-1, 1, 2)], False, (0, 0, 255), 2)

            last = est_points_xyz[-1]
            cv2.putText(
                traj_vis,
                f"last xyz (aligned): [{last[0]:.3f}, {last[1]:.3f}, {last[2]:.3f}]",
                (15, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                2,
            )

        cv2.putText(
            traj_vis,
            "red: estimate  blue: ground truth",
            (15, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
        )

    def view(self):
        image = None
        camera = None
        trajectory = DynamicArray()

        cv2.namedWindow("VIO Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("VIO Trajectory", cv2.WINDOW_NORMAL)

        while True:
            camera, image = self._drain_queues(camera, trajectory, image)

            if image is None:
                cam_vis = np.zeros((480, 752, 3), dtype=np.uint8)
                cv2.putText(
                    cam_vis,
                    "Waiting for image...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
            else:
                cam_vis = image

            traj_vis = np.ones((600, 600, 3), dtype=np.uint8) * 255
            cv2.putText(
                traj_vis,
                f"poses: {self.pose_updates}",
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                2,
            )
            cv2.putText(
                traj_vis,
                f"images: {self.image_updates}",
                (15, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                2,
            )

            points_xyz = trajectory.array()
            self._draw_trajectory(traj_vis, points_xyz, self.gt_points_xyz)

            cv2.imshow("VIO Camera", cam_vis)
            cv2.imshow("VIO Trajectory", traj_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            time.sleep(0.005)

        cv2.destroyWindow("VIO Camera")
        cv2.destroyWindow("VIO Trajectory")


class DynamicArray(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)

        self.data = np.zeros((1000, *shape))
        self.shape = shape
        self.ind = 0

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])

    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize((2 * len(self.data), *self.shape), refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind : self.ind + len(xs)] = xs
        else:
            for i, x in enumerate(xs):
                self.data[self.ind + i] = x
        self.ind += len(xs)

    def array(self):
        return self.data[: self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[: self.ind]:
            yield x
