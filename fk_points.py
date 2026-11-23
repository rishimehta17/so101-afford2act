"""The 4 gripper points from the measured joints (replaces the ArUco marker + CSRT; user decision 2026-11-16).

joints (deg, my_follower_arm_v5) + gripper reading (0-100)
  -> + fitted joint zero offsets, jaw angle = a * gripper + b
  -> forward kinematics to the URDF link moving_jaw_so101_v1_link (the marker sat on the moving jaw)
  -> the 4 fitted marker corners, in the robot base frame
  -> camera frame (R, t) -> pixels (K)
  -> axis-aligned box of the 4 corners (TL, TR, BR, BL), the same format the paper's code made from the marker.
Calibration: calib/camera_robot.json, fitted by 4_fit_camera.py from the demos (held-out demos: median 2.5-3.4 px).
Used by 5_fk_points.py (training data) and 8_run_robot.py (robot), so both see the same points.
"""
import json
from pathlib import Path

import numpy as np

from common import ARM, HERE, URDF

CALIB = HERE / "calib/camera_robot.json"


class FKPoints:
    def __init__(self, calib=CALIB):
        from lerobot.model.kinematics import RobotKinematics
        c = json.loads(Path(calib).read_text())
        self.c = c
        self.K = np.array(c["K"])
        self.R = np.array(c["R_cam_base"])
        self.t = np.array(c["tvec_cam_base_m"])
        self.corners = np.array(c["corners_in_jaw_m"])          # (4,3) in the moving-jaw frame
        self.dq = np.array([c["joint_offsets_deg"][j] for j in ARM])
        self.a, self.b = c["jaw_deg_per_gripper_unit"], c["jaw_deg_offset"]
        self.kin = RobotKinematics(str(URDF), c["frame"], ARM + ["gripper"])

    def corners_px(self, q5, gripper):
        """q5 (5,) arm joints deg, gripper reading 0-100 -> (4,2) px of the 4 marker corners, (4,3) base frame m."""
        T = self.kin.forward_kinematics(np.r_[np.asarray(q5, float) + self.dq, self.a * float(gripper) + self.b])
        pb = self.corners @ T[:3, :3].T + T[:3, 3]
        pc = pb @ self.R.T + self.t
        uv = np.c_[self.K[0, 0] * pc[:, 0] / pc[:, 2] + self.K[0, 2], self.K[1, 1] * pc[:, 1] / pc[:, 2] + self.K[1, 2]]
        return uv, pb

    def box(self, q5, gripper):
        """-> (4,2) px: axis-aligned box of the 4 corners, TL, TR, BR, BL (paper format)."""
        uv, _ = self.corners_px(q5, gripper)
        x0, y0 = uv.min(0)
        x1, y1 = uv.max(0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
