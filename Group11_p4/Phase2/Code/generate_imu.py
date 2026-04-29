import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import os


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


# this calls the different derivative functions above and gets the acceleration and angular velocity
def compute_imu(pos, quat, dt):
    vel = diff_vec(pos, dt)
    acc_world = diff_vec(vel, dt)

    omega_world = diff_quat(quat, dt)

    g = (0,0,-9.81)

    acc_body = []
    omega_body = []

    for i in range(len(pos)):
        q = quat_normalize(quat[i])

        # subtract gravity
        a = tuple(acc_world[i][j] for j in range(3))
        # a = tuple(acc_world[i][j] - g[j] for j in range(3))

        # world -> body
        a_b = quat_rotate(quat_conj(q), a)
        w_b = quat_rotate(quat_conj(q), omega_world[i])

        acc_body.append(a_b)
        omega_body.append(w_b)

    return acc_body, omega_body


# visualization for the data
def save_imu_plot(acc, omega, dt, output_dir, sequence_name):

    t = [i * dt for i in range(len(acc))]

    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

    labels = ["x", "y", "z"]

    # --- Acceleration (top 3) ---
    for i in range(3):
        axs[i].plot(t, [a[i] for a in acc])
        axs[i].set_ylabel(f"a_{labels[i]} (m/s²)")
        axs[i].grid()

    # --- Angular velocity (bottom 3) ---
    for i in range(3):
        axs[i+3].plot(t, [w[i] for w in omega])
        axs[i+3].set_ylabel(f"ω_{labels[i]} (rad/s)")
        axs[i+3].grid()

    axs[-1].set_xlabel("Time (s)")

    fig.suptitle(f"IMU Data – {sequence_name}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(output_dir, f"{sequence_name}_imu_plot.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[PLOT SAVED] {save_path}")

# this is the big file that gets called
def process_file(pose_path, hz):
    dt = 1.0 / hz

    frames, pos, quat = read_poses(pose_path)
    acc, omega = compute_imu(pos, quat, dt)

    seq_dir = pose_path.parent
    seq_name = seq_dir.name

    # Save IMU CSV
    imu_path = seq_dir / "imu.csv"
    write_imu(imu_path, frames, acc, omega, dt)

    # Save plot into Output folder inside sequence
    output_dir = seq_dir
    save_imu_plot(acc, omega, dt, output_dir, seq_name)

    print(f"[OK] {pose_path} -> {imu_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=100)
    args = parser.parse_args()

    # goes through all the poses.csv files in the Data directory
    for p in args.data_root.rglob("poses.csv"):
        process_file(p, args.hz)

if __name__ == "__main__":
    main()