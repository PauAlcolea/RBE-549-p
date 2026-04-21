import pandas as pd

df = pd.read_csv("../Data/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv")

with open("../Output/traj_gt.txt","w") as f:
    for _,row in df.iterrows():
        t = row["#timestamp"] * 1e-9
        px,py,pz = row[" p_RS_R_x [m]"], row[" p_RS_R_y [m]"], row[" p_RS_R_z [m]"]
        qx,qy,qz,qw = row[" q_RS_x []"], row[" q_RS_y []"], row[" q_RS_z []"], row[" q_RS_w []"]

        f.write(f"{t} {px} {py} {pz} {qx} {qy} {qz} {qw}\n")