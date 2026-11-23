"""Step 1: record demonstrations with the SO101 leader / follower, the front RealSense and the wrist camera.

    python 1_record.py --task mug_a2a -n 10        # recordings/mug_a2a/ep_000, ep_001, ...
    dashboard: http://localhost:8770

You move the LEADER arm; the FOLLOWER copies it (mirrored: the leader faces the follower). While a demo records,
every step (10 Hz) saves the camera frame, the wrist camera frame, both arms' joints and the ArUco marker on the
gripper (the marker is only needed once, to fit the camera calibration in step 4). Between demos the follower keeps
copying the leader (nothing saved), so you can reset the scene. At the start and after every demo the follower drives
itself slowly to the home pose and holds it; bring the leader to the same pose (terminal / dashboard show leader -
follower per joint) and teleop resumes by itself.

Type in this terminal (then Enter):
  home:       Enter = take over anyway (the follower walks slowly to the leader)    q = quit
  waiting:    Enter = start the demo    r = discard the last demo (record it again)    q = quit
  recording:  Enter = stop and keep it  r = stop and discard it (record it again)
Dashboard: live camera with the marker; after each demo its quick checks + Replay / Replay 2x buttons.
Discarded demos are moved to <task>/_discarded/, never deleted. Ctrl+C mid-demo keeps what was recorded.

Saved per demo in recordings/<task>/ep_XXX/ (numbers continue after the demos already there):
  N.png          RealSense colour 1280x720, 15 fps (USB 2), the newest frame each step
  wrist/N.jpg    wrist camera 640x480
  states.csv     frame, t, follower_<joint> (measured), command_<joint> (leader, mirrored) in calibration
                 my_follower_arm_v4 (degrees; gripper 0-100)
  marker.csv     ArUco DICT_4X4_50 id 0: found, id, 4 corners (px; -1 if not found)
  meta.json      timing, camera (intrinsics), robot settings, marker seen fraction, wrist repeats / broken frames
The follower's motors may hold another saved calibration: values are converted exactly to/from my_follower_arm_v4;
nothing is ever written into a motor. Settings for this lab's arms and cameras are in the constants below.
"""
import argparse
import csv
import datetime
import json
import select
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import JOINTS, URDF, calib_converter, matching_files  # noqa: E402
from record_view import RecordView, read_marker, read_states  # noqa: E402

MIRRORED = ("shoulder_pan", "wrist_roll")  # rotations that reverse when the leader faces the follower
# the settings the mug_a2a demos were recorded with
ROBOT = {"follower_port": "/dev/ttyACM0", "follower_id": "my_follower_arm_v4", "leader_port": "/dev/ttyACM1",
         "leader_id": "my_leader_arm_v4", "max_step_deg": 10.0, "mirror": True,
         "mirror_center": {"shoulder_pan": -9.8, "wrist_roll": 2.3}}  # in my_follower_arm_v4 / my_leader_arm_v4 degrees
CAMERA = {"serial": "944122073508", "width": 1280, "height": 720, "fps": 15}  # auto exposure, as recorded
WRIST = {"device": "/dev/v4l/by-id/usb-icSpring_icspring_camera_20240307110322-video-index0", "width": 640,
         "height": 480}
MARKER = {"dictionary": "DICT_4X4_50", "id": 0}
FPS, SECONDS, NUM_EPISODES = 10, 45, 10
HOME_POSE = [-11.6, -104.0, 92.0, 74.4, 3.0, 1.0]  # median start pose of the demos (my_follower_arm_v4)
HANDOVER_DEG, HANDOVER_GRIPPER = 8.0, 15.0
GRIP_THRESH = 5.0  # leader gripper below this = closed


def typed():
    """The line typed in the terminal since the last call, or None (doesn't wait)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        return line.strip().lower() if line else None  # '' = no terminal attached
    return None


# ---------------------------------------------------------------- hardware
class Arms:
    """Follower + leader SO101. Joints in degrees (my_follower_arm_v4 / my_leader_arm_v4), gripper 0-100."""

    def __init__(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        data_id = ROBOT["follower_id"]
        probe = SO101Follower(SO101FollowerConfig(port=ROBOT["follower_port"], id=data_id))
        try:
            found = matching_files(probe)
        except RuntimeError as e:
            sys.exit(f"no follower motors answer on {ROBOT['follower_port']}: is its power supply on?\n{e}")
        if not found:
            sys.exit("no saved calibration matches the follower's motors (see README: arm calibration)")
        self.motor_id = data_id if data_id in found else found[0]
        self.to_data = calib_converter(self.motor_id, data_id)
        self.to_motor = calib_converter(data_id, self.motor_id)
        self.leader = SO101Leader(SO101LeaderConfig(port=ROBOT["leader_port"], id=ROBOT["leader_id"]))
        if ROBOT["leader_id"] not in matching_files(self.leader):
            sys.exit(f"The leader doesn't match calibration '{ROBOT['leader_id']}'")
        self.follower = SO101Follower(SO101FollowerConfig(
            port=ROBOT["follower_port"], id=self.motor_id, cameras={},
            max_relative_target=ROBOT["max_step_deg"]))  # safety: each joint moves at most this per command
        self.mirror, self.mirror_center = ROBOT["mirror"], ROBOT["mirror_center"]
        for arm in (self.leader, self.follower):
            arm.connect(calibrate=False)  # never start the interactive calibration, never write calibration

    def read_leader(self):
        a = self.leader.get_action()
        return np.array([a[f"{j}.pos"] for j in JOINTS])

    def leader_command(self):
        """The leader's pose as a follower command (my_follower_arm_v4). With mirror on, pan and wrist roll are
        reflected around mirror_center, so the follower moves like your reflection in a mirror."""
        q = self.read_leader()
        if self.mirror:
            for j in MIRRORED:
                k = JOINTS.index(j)
                q[k] = 2 * self.mirror_center.get(j, 0.0) - q[k]
        return q

    def read_follower(self):
        o = self.follower.bus.sync_read("Present_Position")
        return np.array([self.to_data(j, o[j]) for j in JOINTS])

    def command(self, q):
        self.follower.send_action({f"{j}.pos": float(self.to_motor(j, v)) for j, v in zip(JOINTS, q)})

    def home(self, q_goal, deg_per_step=1.0, hz=30):
        """Slow, blocking move to q_goal."""
        q0 = self.read_follower()
        steps = max(1, int(np.abs(np.asarray(q_goal) - q0).max() / deg_per_step))
        for q in np.linspace(q0, q_goal, steps + 1)[1:]:
            self.command(q)
            time.sleep(1.0 / hz)

    def close(self):
        print("[arms] disconnecting: the follower goes limp now, support it.")
        for arm in (self.follower, self.leader):
            if arm is not None and arm.is_connected:
                arm.disconnect()


class RealSense:
    """Environment camera: newest colour frame on each read()."""

    def __init__(self, cam):
        import pyrealsense2 as rs
        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(str(cam["serial"]))
        cfg.enable_stream(rs.stream.color, cam["width"], cam["height"], rs.format.bgr8, cam["fps"])
        try:
            profile = self.pipe.start(cfg)
        except RuntimeError as e:
            sys.exit(f"RealSense did not start ({e}). Is it plugged in, and is no other program using it?")
        dev = profile.get_device()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.info = {"serial": dev.get_info(rs.camera_info.serial_number),
                     "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
                     "width": cam["width"], "height": cam["height"], "fps": cam["fps"],
                     "K": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], "dist": list(intr.coeffs),
                     "manual_settings": "auto"}
        for _ in range(30):  # let auto-exposure settle
            self.pipe.wait_for_frames()

    def read(self):
        frames = self.pipe.wait_for_frames(timeout_ms=3000)
        newer = self.pipe.poll_for_frames()
        while newer:  # drop frames that queued up while we were busy
            frames, newer = newer, self.pipe.poll_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data()).copy()

    def close(self):
        self.pipe.stop()


class WristCamera:
    """USB camera on the gripper. A background thread keeps only the newest frame. Frames come as raw MJPEG and are
    checked before decoding: one the USB transfer cut short is skipped (the previous good one stays) and counted in
    info["corrupt"]. read() counts reads that got the same frame as the read before (info["repeats"])."""

    def __init__(self, cam):
        dev = cam["device"]
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f"Wrist camera {dev} did not open. Is it plugged in?")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # raw MJPEG bytes: check each frame is whole before decoding
        self.corrupt, frame = 0, None
        for _ in range(30):
            ok, buf = self.cap.read()
            frame = self._decode(buf) if ok else None
            if frame is not None:
                break
        if frame is None:
            sys.exit(f"Wrist camera {dev} opened but gives no frames")
        self.frame, self.n, self.n_read, self.repeats = frame, 1, 0, 0
        self.lock, self.running = threading.Lock(), True
        self.info = {"device": str(dev), "width": frame.shape[1], "height": frame.shape[0],
                     "fps": round(self.cap.get(cv2.CAP_PROP_FPS), 1)}
        self.thread = threading.Thread(target=self._grab, daemon=True)
        self.thread.start()

    @staticmethod
    def _decode(buf):
        """Raw MJPEG -> BGR image, or None if the frame is incomplete (no end marker, or bottom rows left flat)."""
        if buf is None:
            return None
        if buf.ndim == 3:  # this OpenCV build decoded it already
            return buf
        b = buf.tobytes()
        if len(b) < 1000 or b[:2] != b"\xff\xd8" or b"\xff\xd9" not in b[-32:]:
            return None
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img is None or (img[-16:].reshape(16, -1).std(1) < 1.0).all():  # missing data decodes as flat rows
            return None
        return img

    def _grab(self):
        while self.running:
            ok, buf = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = self._decode(buf)
            with self.lock:
                if frame is None:
                    self.corrupt += 1
                else:
                    self.frame, self.n = frame, self.n + 1

    def read(self):
        with self.lock:
            self.repeats += self.n == self.n_read
            self.n_read = self.n
            return self.frame.copy()

    def peek(self):
        """Newest frame for the dashboard (not counted as a read)."""
        with self.lock:
            return self.frame.copy()

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class MarkerDetector:
    def __init__(self, dictionary="DICT_4X4_50", marker_id=None):
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # sub-pixel corners
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), p)
        self.dictionary, self.marker_id = dictionary, marker_id

    def detect_all(self, frame):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return []
        out = [(int(i), c.reshape(4, 2).astype(np.float64)) for i, c in zip(ids.ravel(), corners)]
        return [o for o in out if self.marker_id is None or o[0] == self.marker_id]

    def detect(self, frame):
        """-> (id, corners (4, 2)) of the marker (the largest if several), or None."""
        found = self.detect_all(frame)
        if not found:
            return None
        return max(found, key=lambda o: cv2.contourArea(o[1].astype(np.float32)))


class EpisodeWriter:
    """Writes frames in background threads so the control loop keeps its rate."""

    def __init__(self, folder):
        self.dir = Path(folder)
        (self.dir / "wrist").mkdir(parents=True)
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.f = open(self.dir / "states.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["frame", "t"] + [f"follower_{j}" for j in JOINTS] + [f"command_{j}" for j in JOINTS])
        self.mf = open(self.dir / "marker.csv", "w", newline="")
        self.mw = csv.writer(self.mf)
        self.mw.writerow(["frame", "found", "id"] + [f"c{k}_{a}" for k in range(4) for a in "xy"])
        self.n = 0

    def add(self, frame, t, follower, command, wrist):
        self.n += 1
        self.pool.submit(cv2.imwrite, str(self.dir / f"{self.n}.png"), frame)
        self.pool.submit(cv2.imwrite, str(self.dir / "wrist" / f"{self.n}.jpg"), wrist, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.w.writerow([self.n, f"{t:.4f}"] + [f"{v:.3f}" for v in follower] + [f"{v:.3f}" for v in command])
        return self.n

    def add_marker(self, found):
        """found = (id, corners (4, 2)) or None, for the step just added."""
        if found is None:
            self.mw.writerow([self.n, 0, -1] + [-1] * 8)
        else:
            self.mw.writerow([self.n, 1, found[0]] + [f"{v:.2f}" for v in np.asarray(found[1]).ravel()])

    def close(self, meta):
        self.pool.shutdown(wait=True)
        self.f.close()
        self.mf.close()
        meta = {"frames": self.n, "command": "leader", **meta}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=lambda o: o.item() if hasattr(o, "item")
                                                       else str(o)))


# ---------------------------------------------------------------- checks (the original's quick_check)
def quick_check(ep, kin, grip_thresh=GRIP_THRESH, max_seconds=None):
    """Checks of a just-recorded demo -> [(ok, text)]: rest at start / end, one grasp, jaw while closed, pause before
    closing, arm still while closing, lift after the jaw shut, follower lag, marker, wrist camera. Rough: to decide
    on the spot whether to redo it."""
    st = read_states(ep)
    meta = json.loads((Path(ep) / "meta.json").read_text())
    t, f, c = st["t"], st["follower"], st["command"]
    dur, late = float(t[-1]), int(meta.get("late_steps", 0))
    hit = bool(max_seconds and dur >= max_seconds - 0.2)
    out = [(late <= 0.05 * len(t) and not hit,
            f"{dur:.1f} s, {len(t)} steps, {late} late" + ("  - hit the time limit: finished?" if hit else ""))]
    out.append((f[0, 1] < -95, f"starts at rest (shoulder_lift {f[0, 1]:.0f})" if f[0, 1] < -95
                else f"does NOT start at rest (shoulder_lift {f[0, 1]:.0f})"))
    end_ok = f[-1, 1] < -95 and c[-1, 5] < grip_thresh
    out.append((end_ok, "ends at rest, gripper closed" if end_ok else
                f"does NOT end at rest with the gripper closed (shoulder_lift {f[-1, 1]:.0f}, gripper {c[-1, 5]:.0f})"))
    closed = c[:, 5] < grip_thresh
    segs, s0 = [], 0
    for i in range(1, len(closed) + 1):
        if i == len(closed) or closed[i] != closed[s0]:
            segs.append((bool(closed[s0]), s0, i))
            s0 = i
    grasps = [(a, b) for k, (cl, a, b) in enumerate(segs) if cl and 0 < k < len(segs) - 1]
    out.append((len(grasps) == 1, " > ".join("closed" if cl else "open" for cl, _, _ in segs)
                + f"  ({len(grasps)} grasp{'s' if len(grasps) != 1 else ''})"))
    if grasps:
        a, b = grasps[0]
        jaw = float(np.median(f[a + 3:b, 5])) if b - a > 4 else float(f[a, 5])
        out.append((1.5 < jaw < 8, f"jaw while closed {jaw:.1f}: " + (
            "holding the rim" if 1.5 < jaw < 8 else
            "closed fully - missed the mug, or a thin rim?" if jaw <= 1.5 else "not squeezed shut?")))
        lvl, s0 = np.median(c[max(0, a - 30):a, 5]), a
        while s0 > 0 and c[s0 - 1, 5] < lvl - 5:  # where you started closing
            s0 -= 1
        speed = np.r_[0, np.abs(np.diff(f[:, :5], axis=0)).max(1)] * float(meta.get("fps_target", FPS))  # deg/s
        k = s0
        while k > 0 and speed[k] < 8:
            k -= 1
        out.append((t[s0] - t[k] <= 1.5, f"still for {t[s0] - t[k]:.1f} s on the rim before you started closing"
                    + ("" if t[s0] - t[k] <= 1.5 else "  - close right away once on the rim")))
        k = a  # the arm must stay still while the jaw closes
        while k + 2 < len(f) and abs(f[k + 2, 5] - f[k, 5]) > 1.0:  # until the follower jaw stops moving
            k += 1
        moved = float(np.abs(f[k, :5] - f[s0, :5]).max())
        z = np.array([kin.forward_kinematics(x[:5])[2, 3] for x in f[k:b]])  # lift: tip rises > 1 cm after the jaw shut
        up = np.where(z > z[0] + 0.01)[0] if len(z) else []
        if len(up):
            lag = float(t[k + up[0]] - t[k])
            out.append((lag <= 1.5, f"lift started {lag:.1f} s after the jaw shut" + ("" if lag <= 1.5 else
                                                                                     "  - lift sooner (count one-two)")))
        out.append((moved <= 4.0, f"arm still while the jaw closed (moved {moved:.1f} deg in {t[k] - t[s0]:.1f} s)"
                    if moved <= 4.0 else f"arm moved {moved:.0f} deg while the jaw was closing - hold still until it is shut"))
    lag = np.percentile(np.abs(f[:, :5] - c[:, :5]).max(1), 95)
    out.append((lag < 30, f"follower lag (95th percentile) {lag:.0f} deg" + ("" if lag < 30 else "  - moving fast")))
    mk = read_marker(ep)
    if mk:
        seen = np.array([mk.get(int(fid)) is not None for fid in st["frames"]])
        msg, ok = f"marker seen in {100 * seen.mean():.0f}% of the steps", seen.mean() >= 0.8
        if grasps:
            near = (t >= t[grasps[0][0]] - 2.0) & (t <= t[grasps[0][0]] + 2.0)
            msg += f", {100 * seen[near].mean():.0f}% around the grasp (+-2 s)"
            ok = ok and seen[near].mean() >= 0.8
        out.append((ok, msg + ("" if ok else "  - marker hidden / turned away: check its placement")))
    wm = meta.get("wrist_camera") or {}
    nw, rep, bad = len(list((Path(ep) / "wrist").glob("*.jpg"))), int(wm.get("repeats", 0)), int(wm.get("corrupt", 0))
    ok = nw == len(t) and rep <= 0.2 * len(t)
    out.append((ok, f"wrist camera: {nw}/{len(t)} frames" + (f", {rep} repeated" if rep else "")
                + (f", {bad} broken frames skipped" if bad else "")
                + ("" if ok else "  - wrist camera dropped frames (cable? USB?)")))
    return [(bool(o), s) for o, s in out]


# ---------------------------------------------------------------- session
def teleop_until_typed(arms, period, cam, live, status):
    """The follower copies the leader (nothing saved) until a line is typed; returns that line."""
    while True:
        t0 = time.perf_counter()
        arms.command(arms.leader_command())
        live.update(cam.read(), status, (120, 255, 120))
        key = typed()
        if key is not None:
            return key
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def slow_handover(arms, deg_per_step=1.0, hz=30, done_deg=2.0):
    """Move the follower slowly to wherever the leader is (no jump); returns when it has caught up."""
    q = arms.read_follower()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 15:
        d = arms.leader_command() - q
        if np.abs(d).max() < done_deg:
            return
        q = q + np.clip(d, -deg_per_step, deg_per_step)
        arms.command(q)
        time.sleep(1.0 / hz)


def home_and_handover(arms, home, period, cam, live, tol=HANDOVER_DEG, grip_tol=HANDOVER_GRIPPER):
    """The follower drives itself slowly to the home pose and holds it. Bring the leader to the same pose: when every
    joint is within tol (gripper grip_tol), teleop resumes. Enter = take over anyway: the follower then moves slowly
    to the leader. Returns "q" if q was typed."""
    print("\n[home] the follower is going home on its own: keep clear of it")
    live.update(cam.read(), "GOING HOME", (0, 200, 255))
    arms.home(home)
    print("[home] now put the leader in the same pose as the follower (leader - follower, deg; ok = close enough).\n"
          "       Enter = take over anyway (the follower moves slowly to the leader), q = quit")
    names = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
    limits = np.array([tol] * 5 + [grip_tol])
    shown, t_print = "", 0.0
    while True:
        t0 = time.perf_counter()
        arms.command(home)  # hold the home pose
        diff = arms.leader_command() - home
        ok = np.abs(diff) < limits
        if ok.all():
            print("\n[home] the leader matches: the follower copies it again.")
            return None
        line = "  ".join(f"{n} {d:+.0f}{' ok' if k else ''}" for n, d, k in zip(names, diff, ok))
        off = ", ".join(f"{n} {d:+.0f}" for n, d, k in zip(names, diff, ok) if not k)
        live.update(cam.read(), f"HOME: match the leader to the follower: {off} (Enter = take over)", (0, 200, 255))
        if line != shown and time.perf_counter() - t_print > 0.3:  # a live line in the terminal
            print(f"\r[home] {line}      ", end="", flush=True)
            shown, t_print = line, time.perf_counter()
        key = typed()
        if key == "q":
            print()
            return "q"
        if key is not None:  # Enter (or anything else): take over without matching
            print("\n[home] taking over: the follower moves slowly to the leader...")
            slow_handover(arms)
            print("[home] done: the follower copies the leader again.")
            return None
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def task_dir_of(task):
    return HERE / "recordings" / task


def next_episode_dir(task):
    """ep_000, ep_001, ...: one past the highest number already there (discarded demos leave gaps)."""
    d = task_dir_of(task)
    nums = [int(p.name[3:]) for p in d.glob("ep_*") if p.is_dir() and p.name[3:].isdigit()] if d.is_dir() else []
    return d / f"ep_{max(nums) + 1 if nums else 0:03d}"


def discard(ep):
    dst = ep.parent / "_discarded" / f"{ep.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    dst.parent.mkdir(exist_ok=True)
    shutil.move(str(ep), str(dst))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mug_a2a")
    ap.add_argument("-n", "--num_episodes", type=int, default=NUM_EPISODES)
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--view_port", type=int, default=8770)
    ap.add_argument("--no_home", action="store_true", help="don't send the follower home after each demo")
    args = ap.parse_args()
    if not sys.stdin.isatty():
        sys.exit("run this in a terminal (it reads Enter / r / q)")
    period = 1.0 / FPS
    task_dir = task_dir_of(args.task)
    task_dir.mkdir(parents=True, exist_ok=True)

    cam = RealSense(CAMERA)  # camera first: fail before the arms are powered if it is missing
    print(f"[cam] RealSense #{cam.info['serial']} USB {cam.info['usb']}, settings {cam.info['manual_settings']}")
    wrist = WristCamera(WRIST)
    print(f"[cam] wrist camera {wrist.info['width']}x{wrist.info['height']} @ {wrist.info['fps']} fps: saved as wrist/N.jpg")
    marker = MarkerDetector(MARKER["dictionary"], MARKER["id"])
    print(f"[marker] looking for ArUco {marker.dictionary} id {marker.marker_id} in every frame: saved as marker.csv")
    arms = Arms()
    print(f"[arms] connected (follower motors: {arms.motor_id}; saved in {ROBOT['follower_id']}): the follower now "
          "copies the leader like a mirror (shoulder_pan and wrist_roll flipped).")
    from lerobot.model.kinematics import RobotKinematics
    kin = RobotKinematics(str(URDF), "gripper_frame_link", JOINTS[:5])
    live = RecordView(args.view_port, grip_thresh=GRIP_THRESH, wrist_cam=wrist, marker_det=marker)
    home = np.array(HOME_POSE, float)
    kept, writer, need_home = [], None, not args.no_home  # kept = demos saved in this session
    t_start, rep0, bad0 = 0.0, 0, 0
    try:
        while len(kept) < args.num_episodes:
            if need_home:  # at the start and after every demo: same start pose each time
                need_home = False
                if home_and_handover(arms, home, period, cam, live) == "q":
                    break
            ep = next_episode_dir(args.task)
            redo = f"   r = discard {kept[-1].name} and redo it" if kept else ""
            print(f"\n[{len(kept) + 1}/{args.num_episodes}] Reset the scene, hold the leader still at the rest pose.\n"
                  f"      Enter = record {ep.name}{redo}   q = quit")
            key = teleop_until_typed(arms, period, cam, live, f"READY {ep.name}: place the mug, then Enter")
            if key == "q":
                break
            if key == "r":
                if kept:
                    gone = kept.pop()
                    print(f"[rec] discarded {gone.name} -> {discard(gone).relative_to(task_dir)}; it will be recorded again")
                    live.set_last(None, [])
                else:
                    print("[rec] no demo from this session to redo")
                continue

            writer = EpisodeWriter(ep)
            n_seen = 0
            print(f"[rec] RECORDING {ep.name}   Enter = stop and keep   r = stop and throw away   (max {args.seconds:.0f} s)")
            t_start, late, key, n = time.perf_counter(), 0, None, 0
            rep0, bad0 = wrist.repeats, wrist.corrupt  # counts before this demo
            while True:
                t0 = time.perf_counter()
                command = arms.leader_command()  # what you tell the robot to do (mirrored)
                arms.command(command)
                follower = arms.read_follower()  # where the robot actually is
                frame = cam.read()
                wframe = wrist.read()
                n = writer.add(frame, t0 - t_start, follower, command, wframe)
                found = marker.detect(frame)
                writer.add_marker(found)
                n_seen += found is not None
                if n % 2 == 0:
                    live.update(frame, f"REC {ep.name}  {t0 - t_start:4.1f} s  marker {100 * n_seen / n:.0f}%",
                                (60, 60, 255), wframe, markers=[found] if found is not None else [])
                if n % FPS == 0:
                    print(f"\r[rec] {t0 - t_start:5.1f} s  {n} frames ", end="", flush=True)
                key = typed()
                if key is not None or t0 - t_start >= args.seconds:
                    break
                dt = time.perf_counter() - t0
                late += dt > period
                time.sleep(max(0.0, period - dt))
            dur = time.perf_counter() - t_start
            writer.close({"task": args.task, "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
                          "duration_s": round(dur, 3), "fps_target": FPS, "fps_achieved": round(n / dur, 2),
                          "late_steps": int(late), "env_camera": cam.info,
                          "robot": dict(ROBOT, motor_calibration=arms.motor_id),
                          "marker": {"dictionary": marker.dictionary, "id": marker.marker_id,
                                     "seen_fraction": round(n_seen / max(n, 1), 3)},
                          "wrist_camera": dict(wrist.info, repeats=wrist.repeats - rep0, corrupt=wrist.corrupt - bad0),
                          "recorder": "so101-afford2act/1_record.py"})
            writer = None
            need_home = not args.no_home
            if key == "r":
                print(f"\n[rec] discarded {ep.name} -> {discard(ep).relative_to(task_dir)}; record it again")
                continue
            kept.append(ep)
            print(f"\n[rec] saved {ep.name}: {n} frames, {dur:.1f} s, {n / dur:.1f} Hz"
                  + ("   (hit the time limit: was the demo finished?)" if dur >= args.seconds else ""))
            try:  # a bug in a check must never end the session
                checks = quick_check(ep, kin, GRIP_THRESH, args.seconds)
            except Exception as e:  # noqa: BLE001
                checks = [(False, f"checks failed: {type(e).__name__}: {e}")]
            for ok, text in checks:
                print(f"      {'ok  ' if ok else 'CHECK'} {text}")
            live.set_last(ep, checks)
            print(f"      replay it on the dashboard (http://localhost:{args.view_port}); r = redo it")
        if need_home:  # the session ended right after a demo: leave the arm at home before it goes limp
            print("\n[home] the follower is going home on its own: keep clear of it")
            arms.home(home)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:  # Ctrl+C mid-demo: keep what was recorded
            dur = time.perf_counter() - t_start
            writer.close({"task": args.task, "interrupted": True, "duration_s": round(dur, 3), "fps_target": FPS,
                          "fps_achieved": round(writer.n / max(dur, 1e-6), 2), "env_camera": cam.info,
                          "robot": dict(ROBOT, motor_calibration=arms.motor_id),
                          "wrist_camera": dict(wrist.info, repeats=wrist.repeats - rep0, corrupt=wrist.corrupt - bad0),
                          "recorder": "so101-afford2act/1_record.py"})
            kept.append(writer.dir)
            print(f"\n[rec] {writer.dir.name} was interrupted with Ctrl+C; kept. Discard it if it is bad.")
        print(f"\n[rec] this session: {len(kept)} demo(s) {', '.join(e.name for e in kept)} in {task_dir}")
        live.close()
        cam.close()
        wrist.close()
        arms.close()

    n_eps = len(list(task_dir.glob("ep_*/meta.json")))
    if n_eps:  # processing is done once, by hand, when all demos are recorded
        print(f"\n[data] {n_eps} demos of {args.task} so far. When all are done, continue with step 2:\n"
              f"  python 2_process.py --src recordings/{args.task} --out data/{args.task}")


if __name__ == "__main__":
    main()
