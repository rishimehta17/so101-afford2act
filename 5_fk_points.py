"""Step 5: the 4 gripper points from the recorded joints, and the policy input per frame.

    python 5_fk_points.py --data data/mug_a2a

For every ep_* in --data (steps 2 and 3 done):
  ee_coords.csv      frame, ee_x1..ee_y4 (px): box of the 4 marker corners computed from the follower's measured joints
                     (fk_points.py + calib/camera_robot.json; TL, TR, BR, BL, the format of the paper's code), gz
  merged_coords.csv  frame, ee_x1..ee_y4, gz, obj_x1..obj_y15 = the policy input of each frame (19 points)
No marker or box tracker is involved: the gripper points are never hidden and do not drift.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ARM, HERE
from fk_points import FKPoints

EE_COLS = [f"ee_{a}{i}" for i in range(1, 5) for a in "xy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "data/mug_a2a"))
    a = p.parse_args()
    fk = FKPoints()
    n = 0
    for ep in sorted(Path(a.data).glob("ep_*")):
        if not (ep / "tracks.csv").exists():
            print(f"{ep.name}: no tracks.csv (run 3_track.py) -> skipped")
            continue
        rs = pd.read_csv(ep / "robot_states.csv")
        box = np.stack([fk.box(q, g) for q, g in zip(rs[[f"follower_{j}" for j in ARM]].values, rs.follower_gripper)])
        ee = pd.DataFrame(box.reshape(len(rs), 8).round(2), columns=EE_COLS)
        ee.insert(0, "frame", rs.frame.values)
        ee["gz"] = rs.gz.values
        ee.to_csv(ep / "ee_coords.csv", index=False)
        tracks = pd.read_csv(ep / "tracks.csv")
        assert list(tracks.frame) == list(ee.frame), f"{ep.name}: frame numbers differ"
        ee.merge(tracks, on="frame", how="left").to_csv(ep / "merged_coords.csv", index=False)
        n += 1
        print(f"{ep.name}: {len(rs)} frames -> ee_coords.csv, merged_coords.csv")
    print(f"{n} demos done")


if __name__ == "__main__":
    main()
