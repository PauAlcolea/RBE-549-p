
from queue import Queue
from threading import Lock, Thread

import numpy as np

from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF



class VIO(object):
    def __init__(self, config, img_queue, imu_queue, viewer=None):
        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        self.result_lock = Lock()
        self.estimate_timestamps = []
        self.estimate_positions = []

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            # print('img_msg', img_msg.timestamp)

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.stareo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return
            # print('imu_msg', imu_msg.timestamp)

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                return
            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)

            if result is not None:
                body_t = np.asarray(result.pose.matrix()[:3, 3], dtype=np.float64)
                with self.result_lock:
                    self.estimate_timestamps.append(float(result.timestamp))
                    self.estimate_positions.append(body_t)

                if self.viewer is not None:
                    self.viewer.update_pose(result.cam0_pose)

    def get_estimate_trajectory(self):
        with self.result_lock:
            if len(self.estimate_positions) == 0:
                return np.array([], dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
            return (
                np.asarray(self.estimate_timestamps, dtype=np.float64),
                np.asarray(self.estimate_positions, dtype=np.float64),
            )


def load_groundtruth_trajectory(groundtruth_reader):
    timestamps = []
    positions = []
    for msg in groundtruth_reader:
        timestamps.append(float(msg.timestamp))
        positions.append(np.asarray(msg.p, dtype=np.float64))

    if len(positions) == 0:
        return np.array([], dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.asarray(timestamps, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def align_gt_to_estimate(est_t, est_p, gt_t, gt_p):
    if len(est_t) == 0 or len(gt_t) < 2:
        return np.array([], dtype=np.float64), np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    valid = (est_t >= gt_t[0]) & (est_t <= gt_t[-1])
    if not np.any(valid):
        return np.array([], dtype=np.float64), np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    t = est_t[valid]
    est = est_p[valid]
    gt_interp = np.column_stack([
        np.interp(t, gt_t, gt_p[:, 0]),
        np.interp(t, gt_t, gt_p[:, 1]),
        np.interp(t, gt_t, gt_p[:, 2]),
    ])
    return t, est, gt_interp


def align_se3(est, gt):
    """Rigidly align estimate points to GT points with no scale change."""
    if len(est) == 0 or len(gt) == 0:
        return est, np.eye(3), np.zeros(3)

    est_mean = np.mean(est, axis=0)
    gt_mean = np.mean(gt, axis=0)

    est_centered = est - est_mean
    gt_centered = gt - gt_mean

    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = gt_mean - R @ est_mean
    est_aligned = (R @ est.T).T + t
    return est_aligned, R, t


def save_estimate_csv(output_path, timestamps, positions):
    data = np.column_stack([timestamps, positions])
    header = 'timestamp,p_x,p_y,p_z'
    np.savetxt(output_path, data, delimiter=',', header=header, comments='')


def plot_estimate_vs_groundtruth(t, est, gt, output_path=None, show_plot=True):
    import matplotlib.pyplot as plt

    gt_color = 'tab:pink'
    est_traj_color = 'tab:blue'
    est_err_color = 'black'

    t0 = t[0]
    t_rel = t - t0

    est_aligned, _, _ = align_se3(est, gt)
    err_aligned = est_aligned - gt
    err_norm_aligned = np.linalg.norm(err_aligned, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Top-down trajectory view in X-Z.
    axes[0].plot(gt[:, 0], gt[:, 2], label='Ground Truth', color=gt_color, linewidth=2.0)
    axes[0].plot(est_aligned[:, 0], est_aligned[:, 2], label='Estimate', color=est_traj_color, linewidth=1.7)
    axes[0].set_title('Trajectory (X-Z)')
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Z [m]')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_rel, err_norm_aligned, color=est_err_color, linestyle='-', label='|e|')
    axes[1].set_title('Position Error Norm vs Time')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('|e| [m]')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    rmse_aligned = np.sqrt(np.mean((est_aligned - gt) ** 2, axis=0))
    fig.suptitle(
        f'Estimate vs Ground Truth RMSE [m]: x={rmse_aligned[0]:.3f}, y={rmse_aligned[1]:.3f}, z={rmse_aligned[2]:.3f}',
        fontsize=11,
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        


if __name__ == '__main__':
    import time
    import sys
    import argparse

    from dataset import EuRoCDataset, DataPublisher

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../Data/MH_01_easy', 
        help='Path of EuRoC MAV dataset.')
    parser.add_argument('--view', action='store_true', help='Show trajectory.')
    parser.add_argument('--live-gt', action='store_true', help='Overlay full ground truth trajectory in live viewer.')
    parser.add_argument('--plot-final', action='store_true', help='Plot estimate vs ground truth after run.')
    parser.add_argument('--plot-save', type=str, default=None, help='Optional output image path for final plot.')
    parser.add_argument('--estimate-save', type=str, default=None, help='Optional CSV output path for estimated trajectory.')
    parser.add_argument('--traj-est-save', type=str, default='../Output/traj_est.txt', help='TUM output path for evo trajectory evaluation.')
    args = parser.parse_args()

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=40.)   # start from static state

    use_main_thread_view = False
    if args.view:
        from viewer import Viewer

        use_main_thread_view = (sys.platform == 'darwin')
        viewer = Viewer(start_process=not use_main_thread_view)

        if args.live_gt and use_main_thread_view:
            _, gt_positions = load_groundtruth_trajectory(dataset.groundtruth)
            viewer.set_groundtruth(gt_positions)
    else:
        viewer = None


    img_queue = Queue()
    imu_queue = Queue()
    # gt_queue = Queue()

    config = ConfigEuRoC()
    msckf_vio = VIO(config, img_queue, imu_queue, viewer=viewer)
    msckf_vio.msckf.start_trajectory_logging(args.traj_est_save)


    duration = float('inf')
    ratio = 0.4  # make it smaller if image processing and MSCKF computation is slow
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, ratio)

    now = time.time()
    try:
        imu_publisher.start(now)
        img_publisher.start(now)

        if use_main_thread_view and viewer is not None:
            viewer.view()
        else:
            img_publisher.publish_thread.join()
            imu_publisher.publish_thread.join()
    finally:
        imu_publisher.stop()
        img_publisher.stop()
        msckf_vio.msckf.stop_trajectory_logging()

        msckf_vio.img_thread.join(timeout=3.0)
        msckf_vio.imu_thread.join(timeout=3.0)
        msckf_vio.vio_thread.join(timeout=3.0)

    est_t, est_p = msckf_vio.get_estimate_trajectory()
    if args.estimate_save and len(est_t) > 0:
        save_estimate_csv(args.estimate_save, est_t, est_p)
        print('Saved estimated trajectory to:', args.estimate_save)

    if args.plot_final:
        gt_t, gt_p = load_groundtruth_trajectory(dataset.groundtruth)
        t_plot, est_plot, gt_plot = align_gt_to_estimate(est_t, est_p, gt_t, gt_p)
        if len(t_plot) == 0:
            print('No overlapping timestamps between estimate and ground truth. Skipping final plot.')
        else:
            plot_estimate_vs_groundtruth(
                t_plot,
                est_plot,
                gt_plot,
                output_path=args.plot_save,
                show_plot=True,
            )
            if args.plot_save:
                print('Saved final plot to:', args.plot_save)