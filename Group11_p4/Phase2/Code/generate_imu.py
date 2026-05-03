import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import os
import numpy as np

from imu_noise_utils import (
    acc_gen,
    gyro_gen,
    vib_from_env,
    accel_low_accuracy,
    accel_mid_accuracy,
    accel_high_accuracy,
    gyro_low_accuracy,
    gyro_mid_accuracy,
    gyro_high_accuracy,
)


# read from teh poses.csv
def read_poses(path):
    frames, pos, quat = [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            frames.append(int(row["frame"]))
            pos.append((float(row["tx"]), float(row["ty"]), float(row["tz"])))
            quat.append((float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])))
    return frames, pos, quat

# make the file with the imu data
def write_imu(path, frames, acc, omega, dt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","t","ax","ay","az","wx","wy","wz"])
        for i in range(len(frames)):
            w.writerow([
                frames[i],
                i*dt,
                *acc[i],
                *omega[i]
            ])




# to make sure that the quaternion has a length of 1
def quat_normalize(q):
    w,x,y,z = q
    n = math.sqrt(w*w+x*x+y*y+z*z)
    return (w/n, x/n, y/n, z/n)

# to invert rotation to go "backwards" framewise
def quat_conj(q):
    w,x,y,z = q
    return (w,-x,-y,-z)

# multiplication
def quat_mul(a,b):
    aw,ax,ay,az = a
    bw,bx,by,bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    )

def quat_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

# rotate a vector v with q
def quat_rotate(q, v):
    """Rotate vector v by quaternion q"""
    qv = (0.0, v[0], v[1], v[2])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]






# get velocity or acceleration from a position or a velocity vector
def diff_vec(samples, dt):
    n = len(samples)
    out = [(0,0,0)]*n
    for i in range(n):
        if i == 0:
            d = [(samples[1][j]-samples[0][j])/dt for j in range(3)]
        elif i == n-1:
            d = [(samples[-1][j]-samples[-2][j])/dt for j in range(3)]
        else:
            d = [(samples[i+1][j]-samples[i-1][j])/(2*dt) for j in range(3)]
        out[i] = tuple(d)
    return out

# quaternion form to axis and angle
def quat_to_rotvec(q):
    w,x,y,z = quat_normalize(q)
    if w < 0:
        w,x,y,z = -w,-x,-y,-z
    angle = 2*math.acos(max(-1,min(1,w)))
    s = math.sqrt(1-w*w)
    if s < 1e-8:
        return (0,0,0)
    return (x/s*angle, y/s*angle, z/s*angle)

# for angular velocity
def diff_quat(quats, dt):
    n = len(quats)
    omega = [(0,0,0)]*n
    for i in range(n):
        if i == 0:
            dq = quat_mul(quat_conj(quats[0]), quats[1])
            rv = quat_to_rotvec(dq)
            omega[i] = tuple(v/dt for v in rv)
        elif i == n-1:
            dq = quat_mul(quat_conj(quats[-2]), quats[-1])
            rv = quat_to_rotvec(dq)
            omega[i] = tuple(v/dt for v in rv)
        else:
            dq = quat_mul(quat_conj(quats[i-1]), quats[i+1])
            rv = quat_to_rotvec(dq)
            omega[i] = tuple(v/(2*dt) for v in rv)
    return omega


def diff_vec_periodic(samples, dt):
    n = len(samples)
    out = [(0,0,0)]*n
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        d = [(samples[ip1][j]-samples[im1][j])/(2*dt) for j in range(3)]
        out[i] = tuple(d)
    return out


def diff_quat_periodic(quats, dt):
    n = len(quats)
    qn = [quat_normalize(q) for q in quats]
    omega = [(0,0,0)]*n
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        q_prev = qn[im1]
        q_next = qn[ip1]
        if quat_dot(q_prev, q_next) < 0:
            q_next = tuple(-v for v in q_next)
        dq = quat_mul(quat_conj(q_prev), q_next)
        rv = quat_to_rotvec(dq)
        omega[i] = tuple(v/(2*dt) for v in rv)
    return omega


def is_closed_loop(pos, quat):
    if len(pos) < 4:
        return False

    dx = pos[-1][0] - pos[0][0]
    dy = pos[-1][1] - pos[0][1]
    dz = pos[-1][2] - pos[0][2]
    pos_gap = math.sqrt(dx*dx + dy*dy + dz*dz)
    if pos_gap > 1e-9:
        return False

    q0 = quat_normalize(quat[0])
    qn = quat_normalize(quat[-1])
    return abs(quat_dot(q0, qn)) > 0.999


def compute_world_kinematics(pos, quat, dt):
    if is_closed_loop(pos, quat):
        # Use periodic derivatives on unique samples to avoid boundary artifacts.
        pos_u = pos[:-1]
        quat_u = quat[:-1]

        vel_u = diff_vec_periodic(pos_u, dt)
        acc_u = diff_vec_periodic(vel_u, dt)
        omega_u = diff_quat_periodic(quat_u, dt)

        acc_world = acc_u + [acc_u[0]]
        omega_world = omega_u + [omega_u[0]]
        return acc_world, omega_world

    vel = diff_vec(pos, dt)
    acc_world = diff_vec(vel, dt)
    omega_world = diff_quat(quat, dt)
    return acc_world, omega_world


def rotate_world_to_body(acc_world, omega_world, quat, fixed_heading=False):
    acc_body = []
    omega_body = []

    q_ref = quat_normalize(quat[0]) if fixed_heading else None

    for i in range(len(acc_world)):
        q = q_ref if fixed_heading else quat_normalize(quat[i])
        a_b = quat_rotate(quat_conj(q), acc_world[i])
        w_b = quat_rotate(quat_conj(q), omega_world[i])
        acc_body.append(a_b)
        omega_body.append(w_b)

    return acc_body, omega_body


# this calls the different derivative functions above and gets the acceleration and angular velocity
# this is the ground truth before the noise
def compute_imu(pos, quat, dt):
    acc_world, omega_world = compute_world_kinematics(pos, quat, dt)
    return rotate_world_to_body(acc_world, omega_world, quat, fixed_heading=False)


# # # use the script from https://github.com/prgumd/Oystersim/blob/master/code/ImuUtils.py to add noise
def get_noise_params(profile):
    profile = profile.lower()
    if profile == "low":
        return accel_low_accuracy, gyro_low_accuracy
    if profile == "mid":
        return accel_mid_accuracy, gyro_mid_accuracy
    if profile == "high":
        return accel_high_accuracy, gyro_high_accuracy
    raise ValueError(f"Unknown noise profile '{profile}'. Use low|mid|high.")


def scale_noise_params(acc_err, gyro_err, noise_scale):
    if noise_scale <= 0:
        raise ValueError(f"noise_scale must be > 0, got {noise_scale}")

    acc_scaled = {
        "b": acc_err["b"],
        "b_drift": acc_err["b_drift"] * noise_scale,
        "b_corr": acc_err["b_corr"],
        "vrw": acc_err["vrw"] * noise_scale,
    }
    gyro_scaled = {
        "b": gyro_err["b"],
        "b_drift": gyro_err["b_drift"] * noise_scale,
        "b_corr": gyro_err["b_corr"],
        "arw": gyro_err["arw"] * noise_scale,
    }
    return acc_scaled, gyro_scaled


def add_imu_noise(acc_gt, omega_gt, hz, profile="mid", seed=None, acc_vib=None, gyro_vib=None, noise_scale=1.0):
    if seed is not None:
        np.random.seed(seed)

    acc_err, gyro_err = get_noise_params(profile)
    acc_err, gyro_err = scale_noise_params(acc_err, gyro_err, noise_scale)

    acc_np = np.asarray(acc_gt, dtype=np.float64)
    omega_np = np.asarray(omega_gt, dtype=np.float64)

    acc_vib_def = vib_from_env(acc_vib, hz) if acc_vib else None
    gyro_vib_def = vib_from_env(gyro_vib, hz) if gyro_vib else None

    acc_noisy = acc_gen(hz, acc_np, acc_err, acc_vib_def)
    omega_noisy = gyro_gen(hz, omega_np, gyro_err, gyro_vib_def)
    return acc_noisy.tolist(), omega_noisy.tolist()


# visualization for the data
def save_imu_plot(acc, omega, dt, output_dir, sequence_name, acc_gt=None, omega_gt=None):

    t = [i * dt for i in range(len(acc))]

    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

    labels = ["x", "y", "z"]

    has_gt = acc_gt is not None and omega_gt is not None

    # Top 3 acceleration
    for i in range(3):
        axs[i].plot(t, [a[i] for a in acc], label="noisy")
        if has_gt:
            axs[i].plot(t, [a[i] for a in acc_gt], "--", label="gt", alpha=0.85)
        axs[i].set_ylabel(f"a_{labels[i]} (m/s²)")
        axs[i].grid()
        axs[i].legend(loc="upper right", fontsize=8)

    # Bottom 3 angular velocity
    for i in range(3):
        axs[i+3].plot(t, [w[i] for w in omega], label="noisy")
        if has_gt:
            axs[i+3].plot(t, [w[i] for w in omega_gt], "--", label="gt", alpha=0.85)
        axs[i+3].set_ylabel(f"ω_{labels[i]} (rad/s)")
        axs[i+3].grid()
        axs[i+3].legend(loc="upper right", fontsize=8)

    axs[-1].set_xlabel("Time (s)")

    fig.suptitle(f"IMU Data – {sequence_name}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(output_dir, f"{sequence_name}_imu_plot.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[PLOT SAVED] {save_path}")


def save_imu_plot_with_name(acc, omega, dt, output_dir, file_name, title, acc_gt=None, omega_gt=None):
    # Remove first/last samples in plots to avoid edge-derivative artifacts that
    # often appear as sharp vertical lines at the boundaries.
    start_idx = 2 if len(acc) > 4 else (1 if len(acc) > 2 else 0)
    end_idx = len(acc) - start_idx if len(acc) > 4 else (len(acc) - 1 if len(acc) > 2 else len(acc))
    t = [i * dt for i in range(start_idx, end_idx)]
    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    labels = ["x", "y", "z"]
    has_gt = acc_gt is not None and omega_gt is not None

    for i in range(3):
        axs[i].plot(t, [a[i] for a in acc[start_idx:end_idx]], label="signal")
        if has_gt:
            axs[i].plot(t, [a[i] for a in acc_gt[start_idx:end_idx]], "--", label="gt", alpha=0.85)
        axs[i].set_ylabel(f"a_{labels[i]} (m/s²)")
        axs[i].grid()
        axs[i].legend(loc="upper right", fontsize=8)

    for i in range(3):
        axs[i+3].plot(t, [w[i] for w in omega[start_idx:end_idx]], label="signal")
        if has_gt:
            axs[i+3].plot(t, [w[i] for w in omega_gt[start_idx:end_idx]], "--", label="gt", alpha=0.85)
        axs[i+3].set_ylabel(f"ω_{labels[i]} (rad/s)")
        axs[i+3].grid()
        axs[i+3].legend(loc="upper right", fontsize=8)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT SAVED] {save_path}")

# this is the big file that gets called
def process_file(pose_path, hz, noise_profile, noise_seed, acc_vib, gyro_vib, noise_scale):
    dt = 1.0 / hz

    frames, pos, quat = read_poses(pose_path)
    acc_world, omega_world = compute_world_kinematics(pos, quat, dt)

    # Tangent-heading body frame (current/default behavior).
    acc_gt, omega_gt = rotate_world_to_body(acc_world, omega_world, quat, fixed_heading=False)
    # Fixed-heading body frame (uses first pose orientation for all frames).
    acc_gt_fixed, omega_gt_fixed = rotate_world_to_body(acc_world, omega_world, quat, fixed_heading=True)

    acc_noisy, omega_noisy = add_imu_noise(
        acc_gt,
        omega_gt,
        hz,
        profile=noise_profile,
        seed=noise_seed,
        acc_vib=acc_vib,
        gyro_vib=gyro_vib,
        noise_scale=noise_scale,
    )
    fixed_seed = None if noise_seed is None else noise_seed + 1
    acc_noisy_fixed, omega_noisy_fixed = add_imu_noise(
        acc_gt_fixed,
        omega_gt_fixed,
        hz,
        profile=noise_profile,
        seed=fixed_seed,
        acc_vib=acc_vib,
        gyro_vib=gyro_vib,
        noise_scale=noise_scale,
    )

    seq_dir = pose_path.parent
    seq_name = seq_dir.name

    # Save GT and noisy IMU CSV
    imu_gt_path = seq_dir / "imu_gt.csv"
    imu_noisy_path = seq_dir / f"{seq_name}_imu.csv"
    imu_gt_fixed_path = seq_dir / "imu_gt_fixed_heading.csv"
    imu_noisy_fixed_path = seq_dir / f"{seq_name}_fixed_heading_imu.csv"
    write_imu(imu_gt_path, frames, acc_gt, omega_gt, dt)
    write_imu(imu_noisy_path, frames, acc_noisy, omega_noisy, dt)
    write_imu(imu_gt_fixed_path, frames, acc_gt_fixed, omega_gt_fixed, dt)
    write_imu(imu_noisy_fixed_path, frames, acc_noisy_fixed, omega_noisy_fixed, dt)

    # Save plot into Output folder inside sequence
    output_dir = seq_dir
    # Existing tangent-heading plot (signal=noisy, dashed=GT tangent heading).
    save_imu_plot_with_name(
        acc_noisy,
        omega_noisy,
        dt,
        output_dir,
        f"{seq_name}_imu_plot.png",
        f"IMU (Tangent Heading Body Frame) - {seq_name}",
        acc_gt=acc_gt,
        omega_gt=omega_gt,
    )
    # Additional fixed-heading body-frame plot (GT only reference plot).
    save_imu_plot_with_name(
        acc_noisy_fixed,
        omega_noisy_fixed,
        dt,
        output_dir,
        f"{seq_name}_imu_fixed_heading_plot.png",
        f"IMU (Fixed Heading Body Frame) - {seq_name}",
        acc_gt=acc_gt_fixed,
        omega_gt=omega_gt_fixed,
    )

    print(
        f"[OK] {pose_path} -> {imu_gt_path}, {imu_noisy_path}, "
        f"{imu_gt_fixed_path}, {imu_noisy_fixed_path}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=100)
    parser.add_argument("--noise-profile", type=str, default="mid", choices=["low", "mid", "high"])
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Scales IMU stochastic noise terms (vrw/arw/b_drift).")
    parser.add_argument("--acc-vib", type=str, default=None, help="e.g. '[0.03 0.01 0.01]-random'")
    parser.add_argument("--gyro-vib", type=str, default=None, help="e.g. '[0.2 0.2 0.1]d-1Hz-sinusoidal'")
    args = parser.parse_args()

    # Prefer the unified dataset layout (poses.csv), fallback for older IMU-only sets.
    pose_files = sorted(args.data_root.rglob("poses.csv"))

    for p in pose_files:
        process_file(
            p,
            args.hz,
            args.noise_profile,
            args.noise_seed,
            args.acc_vib,
            args.gyro_vib,
            args.noise_scale,
        )

if __name__ == "__main__":
    main()
