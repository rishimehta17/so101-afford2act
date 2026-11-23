"""Step 4: fit camera <-> robot from the recorded demos (no extra recording, no robot).

    python 4_fit_camera.py --data data/mug_a2a          # -> calib/camera_robot.json

Known per frame:  follower joints incl. gripper (data/<task>/ep_XXX/robot_states.csv, step 2)
                  marker corners in the image (the raw recording's marker.csv, px, when found)
                  camera intrinsics K (meta.json; distortion 0)
The marker sits on the MOVING JAW (it moves 36 px when only the gripper opens, ep_001 frames 19-26), so the chain
goes all the way to the URDF link moving_jaw_so101_v1_link, through the URDF joint "gripper".
Fitted:  T_cam_base               where the robot base is, seen from the camera (6 numbers)
         4 marker corners          in the moving-jaw frame (4 x 3 numbers)
         jaw angle = a * g + b     g = follower gripper reading (0-100); start a = 131.2 deg / 100 (calibration range)
         5 joint zero offsets      (deg) our joint calibration's zero need not match the URDF's zero
Model:   joints --FK--> T_base_jaw ; p_base = T_base_jaw @ p_jaw ; p_cam = R p_base + t ;
         u = fx * X/Z + cx ,  v = fy * Y/Z + cy
Fit on all demos except --check (default: the 7 demos our calibration was checked on); those are only checked.
Writes calib/camera_robot.json and prints reprojection errors (px). Ours: held-out median 2.5-3.4 px (1.5-1.9 mm).
Refit whenever the camera has moved.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from common import ARM, HERE, URDF

JAW_FRAME = "moving_jaw_so101_v1_link"
G_SCALE = (3535 - 2042) / 4096 * 360 / 100  # deg per gripper unit, from my_follower_arm_v5 range_min/max
CHECK = ["ep_001", "ep_015", "ep_018", "ep_021", "ep_022", "ep_030", "ep_031"]


def kin():
    from lerobot.model.kinematics import RobotKinematics
    return RobotKinematics(str(URDF), JAW_FRAME, ARM + ["gripper"])


def joints_deg(Q, G, p):
    """Q (N,5) arm deg, G (N,) gripper 0-100, p = dict(dq(5), a, b) -> (N,6) deg for the URDF"""
    return np.column_stack([Q + p["dq"], p["a"] * G + p["b"]])


def fk_all(k, Q6):
    return np.stack([k.forward_kinematics(q) for q in Q6])


def load(d):
    """d = data/<task>/<ep> -> Q (N,5), G (N,), marker corners (N,4,2) or nan, frames, K"""
    src = Path(json.loads((d / "params.json").read_text())["source"])
    rs = pd.read_csv(d / "robot_states.csv")
    mk = pd.read_csv(src / "marker.csv").set_index("frame")
    K = np.array(json.loads((src / "meta.json").read_text())["env_camera"]["K"])
    uv = np.full((len(rs), 4, 2), np.nan)
    for i, f in enumerate(rs.frame):
        if f in mk.index and mk.loc[f, "found"] == 1:
            uv[i] = mk.loc[f, [f"c{c}_{a}" for c in range(4) for a in "xy"]].values.reshape(4, 2)
    return rs[[f"follower_{j}" for j in ARM]].values, rs.follower_gripper.values, uv, rs.frame.values, K


def project(cam, pj, T, K):
    """cam = [rvec(3), tvec(3)], pj = (4,3) corners in jaw frame, T (N,4,4) base<-jaw -> px (N,4,2), cam pts, base pts"""
    R = Rotation.from_rotvec(cam[:3]).as_matrix()
    pb = np.einsum("nij,kj->nki", T[:, :3, :3], pj) + T[:, None, :3, 3]
    pc = pb @ R.T + cam[3:6]
    uv = np.stack([K[0, 0] * pc[..., 0] / pc[..., 2] + K[0, 2], K[1, 1] * pc[..., 1] / pc[..., 2] + K[1, 2]], -1)
    return uv, pc, pb


def unpack(z):
    return z[:6], z[6:18].reshape(4, 3), dict(dq=z[18:23], a=z[23], b=z[24])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data/mug_a2a"))
    ap.add_argument("--check", nargs="*", default=CHECK, help="demos kept out of the fit and only checked")
    ap.add_argument("--out", default=str(HERE / "calib/camera_robot.json"))
    args = ap.parse_args()
    eps = sorted(d.name for d in Path(args.data).glob("ep_*") if (d / "robot_states.csv").exists())
    check_eps = [e for e in args.check if e in eps]
    fit_eps = [e for e in eps if e not in check_eps]
    k = kin()
    data = {e: load(Path(args.data) / e) for e in fit_eps + check_eps}
    K = data[fit_eps[0]][4]
    Q = np.concatenate([data[e][0][::2] for e in fit_eps])
    G = np.concatenate([data[e][1][::2] for e in fit_eps])
    U = np.concatenate([data[e][2][::2] for e in fit_eps])
    ok = ~np.isnan(U[:, 0, 0])
    Q, G, U = Q[ok], G[ok], U[ok]
    print(f"fit on {len(fit_eps)} demos, {len(Q)} frames with the marker seen (every 2nd frame)")

    # stage 1: joints as recorded, jaw = +/-G_SCALE * g; fit camera + corners only (FK computed once)
    best = None
    for a in (G_SCALE, -G_SCALE):
        p = dict(dq=np.zeros(5), a=a, b=0.0)
        T = fk_all(k, joints_deg(Q, G, p))
        _, rv, tv, _ = cv2.solvePnPRansac(T[:, :3, 3].astype(np.float64), U.mean(1).astype(np.float64), K, None,
                                          reprojectionError=15, iterationsCount=500)
        x0 = np.concatenate([rv.ravel(), tv.ravel(), np.zeros(12)])
        s = least_squares(lambda x: (project(x[:6], x[6:].reshape(4, 3), T, K)[0] - U).ravel(), x0, loss="huber",
                          f_scale=3.0, x_scale="jac")
        e = np.linalg.norm(project(s.x[:6], s.x[6:].reshape(4, 3), T, K)[0] - U, axis=-1)
        print(f"stage 1, jaw scale {a:+.3f} deg/unit: median {np.median(e):.2f} px, p90 {np.percentile(e, 90):.2f}")
        if best is None or np.median(e) < best[0]:
            best = (np.median(e), np.r_[s.x, np.zeros(5), a, 0.0])

    # stage 2: everything, incl. joint offsets and jaw mapping (every 3rd fit frame to keep it quick)
    Qs, Gs, Us = Q[::3], G[::3], U[::3]
    def res(z):
        cam, pj, p = unpack(z)
        return (project(cam, pj, fk_all(k, joints_deg(Qs, Gs, p)), K)[0] - Us).ravel()
    lo = np.r_[np.full(18, -np.inf), np.full(5, -15.0), -3.0, -60.0]
    hi = np.r_[np.full(18, np.inf), np.full(5, 15.0), 3.0, 60.0]
    sol = least_squares(res, best[1], loss="huber", f_scale=3.0, x_scale="jac", bounds=(lo, hi), diff_step=1e-4)
    cam, pj, p = unpack(sol.x)
    e_fit = np.linalg.norm(project(cam, pj, fk_all(k, joints_deg(Q, G, p)), K)[0] - U, axis=-1)
    print(f"stage 2 (+ joint offsets, jaw mapping): median {np.median(e_fit):.2f} px, p90 {np.percentile(e_fit, 90):.2f}")
    print("joint offsets (deg):", dict(zip(ARM, np.round(p["dq"], 2))), f" jaw deg = {p['a']:.3f} * g + {p['b']:.2f}")

    report = {}
    for e in check_eps:
        Qe, Ge, Ue, _, _ = data[e]
        m = ~np.isnan(Ue[:, 0, 0])
        uv, pc, _ = project(cam, pj, fk_all(k, joints_deg(Qe[m], Ge[m], p)), K)
        err = np.linalg.norm(uv - Ue[m], axis=-1)
        mm = err * pc[..., 2] * 1000 / K[0, 0]
        report[e] = dict(frames=int(m.sum()), median_px=round(float(np.median(err)), 2),
                         p90_px=round(float(np.percentile(err, 90)), 2), median_mm=round(float(np.median(mm)), 2))
        print(f"check {e}: {m.sum():3d} frames  median {np.median(err):.2f} px ({np.median(mm):.1f} mm)  "
              f"p90 {np.percentile(err, 90):.2f} px")

    R = Rotation.from_rotvec(cam[:3]).as_matrix()
    cam_in_base = -R.T @ cam[3:6]
    print("camera position in robot base frame (m):", np.round(cam_in_base, 3))
    print("marker corners in jaw frame (mm):\n", np.round(pj * 1000, 1))
    print("marker side lengths (mm):", np.round([np.linalg.norm(pj[i] - pj[(i + 1) % 4]) * 1000 for i in range(4)], 1))
    out = dict(K=K.tolist(), rvec_cam_base=cam[:3].tolist(), tvec_cam_base_m=cam[3:6].tolist(),
               R_cam_base=R.tolist(), camera_in_base_m=cam_in_base.tolist(),
               corners_in_jaw_m=pj.tolist(), frame=JAW_FRAME, urdf="SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
               joint_offsets_deg=dict(zip(ARM, p["dq"].round(3).tolist())),
               jaw_deg_per_gripper_unit=float(p["a"]), jaw_deg_offset=float(p["b"]),
               joints_calibration="my_follower_arm_v5", fit_demos=fit_eps,
               fit_median_px=float(np.median(e_fit)), fit_p90_px=float(np.percentile(e_fit, 90)), checks=report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    sys.exit(main())
