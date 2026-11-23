"""Step 2: robot states of each recorded demo.

    python 2_process.py --src recordings/mug_a2a --out data/mug_a2a          # all demos
    python 2_process.py --src recordings/mug_a2a --out data/mug_a2a --eps ep_000

Reads (never changes) a raw recording: states.csv (follower joints + leader command, recorded calibration),
meta.json. Writes data/<task>/<ep>/:
  robot_states.csv  frame, t, gripper pose from forward kinematics (ee_x ee_y ee_z m, ee_rx ee_ry ee_rz rotation
                    vector, base frame, URDF frame gripper_frame_link), gz (1 = gripper closed), follower_<joint>,
                    command_<joint> (deg; gripper 0-100), all in the follower's CURRENT calibration (--follower_id),
                    so they match what the robot reports at run time
  params.json       where the images are, calibrations, frame count
  check.png         pose and gripper over time
gz = 1 when the recorded leader gripper command is below 15 (of 0-100).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import ARM, HERE, JOINTS, URDF, calib_converter

GZ_BELOW = 15.0


def process(src, dst, follower_id):
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.utils.rotation import Rotation

    meta = json.loads((src / "meta.json").read_text())
    rows = list(csv.DictReader(open(src / "states.csv")))
    conv = calib_converter(meta["robot"]["follower_id"], follower_id)
    kin = RobotKinematics(str(URDF), "gripper_frame_link", ARM)
    dst.mkdir(parents=True, exist_ok=True)
    states = []
    for r in rows:
        fol = {j: conv(j, float(r[f"follower_{j}"])) for j in JOINTS}
        cmd = {j: conv(j, float(r[f"command_{j}"])) for j in JOINTS}
        T = kin.forward_kinematics(np.array([fol[j] for j in ARM]))
        rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
        gz = int(float(r["command_gripper"]) < GZ_BELOW)  # recorded leader command, before any conversion
        states.append({"frame": int(r["frame"]), "t": float(r["t"]), "ee_x": T[0, 3], "ee_y": T[1, 3],
                       "ee_z": T[2, 3], "ee_rx": rv[0], "ee_ry": rv[1], "ee_rz": rv[2], "gz": gz,
                       **{f"follower_{j}": fol[j] for j in JOINTS}, **{f"command_{j}": cmd[j] for j in JOINTS}})
    with open(dst / "robot_states.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(states[0]))
        w.writeheader()
        for s in states:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in s.items()})
    info = {"source": str(src.resolve()), "frames": len(states), "fps_recorded": meta.get("fps_achieved"),
            "calibration_recorded": meta["robot"]["follower_id"], "calibration_used": follower_id,
            "gz_closed_if_leader_below": GZ_BELOW, "fk_frame": "gripper_frame_link"}
    (dst / "params.json").write_text(json.dumps(info, indent=2))
    check_image(dst, states)
    return info


def check_image(dst, states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([s["t"] for s in states])
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k in ("ee_x", "ee_y", "ee_z"):
        ax[0].plot(t, [100 * s[k] for s in states], label=k + " (cm)")
    ax[0].legend(loc="upper right")
    ax[0].set_title("gripper position from FK (robot base frame)")
    ax[1].plot(t, [s["follower_gripper"] for s in states], label="follower gripper (0-100)")
    ax[1].plot(t, [100 * s["gz"] for s in states], "k--", label="gz x100 (1 = closed)")
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(dst / "check.png", dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(HERE / "recordings/mug_a2a"), help="folder with ep_* recordings")
    p.add_argument("--eps", nargs="*", help="which demos (default: all with a meta.json)")
    p.add_argument("--out", default=str(HERE / "data/mug_a2a"))
    p.add_argument("--follower_id", default="my_follower_arm_v5", help="calibration the robot uses now")
    a = p.parse_args()
    src = Path(a.src)
    eps = a.eps or sorted(d.name for d in src.glob("ep_*") if (d / "meta.json").exists())
    for ep in eps:
        info = process(src / ep, Path(a.out) / ep, a.follower_id)
        print(f"{ep}: {info['frames']} frames -> {Path(a.out) / ep}")


if __name__ == "__main__":
    main()
