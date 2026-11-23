"""Shared paths and SO101 helpers.

Outside dependencies (not part of this repo, see README "Setup"); each path can be set with an environment variable:
  A2A_REPO     the Afford2Act authors' code (the folder that holds OOAL/ and d3_utils/)  default: ../afford2act
  OOAL_CKPT    OOAL checkpoint "seen_best" (OOAL README, Google Drive)                    default: checkpoints/ooal/seen_best
  SO101_URDF   SO101 robot model from TheRobotStudio/SO-ARM100                            default:
               third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
  LeRobot keeps the arms' calibration files in ~/.cache/huggingface/lerobot/calibration/ (LEROBOT_CALIB to change).
"""
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
A2A_REPO = Path(os.environ.get("A2A_REPO", HERE.parent / "afford2act"))
OOAL_CKPT = Path(os.environ.get("OOAL_CKPT", HERE / "checkpoints/ooal/seen_best"))
URDF = Path(os.environ.get("SO101_URDF", HERE / "third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"))
LEROBOT_CALIB = Path(os.environ.get("LEROBOT_CALIB", Path.home() / ".cache/huggingface/lerobot/calibration"))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = JOINTS[:5]


def calib_converter(src_id, dst_id):
    """Joint values read with follower calibration src_id -> the same physical angle under dst_id.
    LeRobot: Present = Actual - Homing_Offset; arm joints deg = (Present - mid) * 360 / 4095 (mid = middle of the
    range); gripper = (Present - min) / (max - min) * 100. Nothing is written into the motors."""
    folder = LEROBOT_CALIB / "robots/so_follower"
    a = json.loads((folder / f"{src_id}.json").read_text())
    b = json.loads((folder / f"{dst_id}.json").read_text())

    def convert(j, v):
        A, B = a[j], b[j]
        if j == "gripper":
            actual = A["range_min"] + v / 100 * (A["range_max"] - A["range_min"]) + A["homing_offset"]
            return (actual - B["homing_offset"] - B["range_min"]) / (B["range_max"] - B["range_min"]) * 100
        mid_a, mid_b = (A["range_min"] + A["range_max"]) / 2, (B["range_min"] + B["range_max"]) / 2
        return v + (mid_a + A["homing_offset"] - B["homing_offset"] - mid_b) * 360 / 4095
    return convert


def connect_bus(arm, tries=3):
    """Open the motor bus only (no torque change). Retries: the Feetech bus sometimes returns one corrupted packet."""
    for i in range(tries):
        try:
            return arm.bus.connect()
        except ConnectionError:
            if i == tries - 1:
                raise
            time.sleep(0.2)


def matching_files(arm):
    """Read the calibration stored inside the motors (read-only) and list every saved file that matches it."""
    connect_bus(arm)
    in_motors = arm.bus.read_calibration()
    arm.bus.disconnect(disable_torque=False)  # leave the motors exactly as they were
    matches = []
    for f in sorted(arm.calibration_dir.glob("*.json")):
        saved = json.loads(f.read_text())
        if all(m in saved and (saved[m]["homing_offset"], saved[m]["range_min"], saved[m]["range_max"])
               == (c.homing_offset, c.range_min, c.range_max) for m, c in in_motors.items()):
            matches.append(f.stem)
    return matches
