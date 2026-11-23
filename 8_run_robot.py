"""Step 8: run the policy on the SO101 (one rollout, or a session of many controlled from this terminal).

    python 8_run_robot.py                                   # DRY RUN: camera + policy, motors only read
    python 8_run_robot.py --execute --session               # many rollouts: READY -> CHECK -> RUN -> REVIEW
    live view: http://localhost:8770

Per rollout:
  start      the arm goes slowly to the demos' start pose; the gripper opens (the demos always open first, and the
             policy waits for the open gripper as its "go" signal)
  1st frame  OOAL "hold" + DINO best-buddy match to references/<task> -> the 15 handle points (aim for ~15)
  each step  measured joints -> 4 gripper points (fk_points.py)          CoTracker (online) -> 15 handle points
             19 points (x, y) now + 5 and 10 steps ago -> policy -> next 10 joint changes + 10 x P(closed)
             temporal ensembling (paper's weights) -> joint target = previous target + change, kept within --leash deg
             of the measured joints and inside the demos' joint range +- 5 deg
             feedback loop: send, then read the joints every 20 ms until every arm joint is within --settle_tol deg
             (at most --settle_max s); a per-joint offset (gain --sag_ki, max +-3 deg) adds back the error left at rest
             (gravity sag). One policy step per loop: the arm sets the pace.
             gripper: P(closed) averaged like the joints; close above --grip_on, open below --grip_off only after
             --grip_min_closed s closed
Session keys (type, then Enter):
  READY   place the mug (the arm holds the start pose)   Enter = check the first frame   q = quit
  CHECK   handle points shown                            Enter = run   c = check again   q = quit
  RUN                                                    Enter = stop early   Ctrl+C = end the session
  REVIEW                                                 Enter = keep   d = discard (moved to _discarded/)   r = replay
Saved in runs/<policy>_[session_]<time>[_dry]/[roll_NNN/]: video.mp4 (camera + graphs), log.csv (commands, measured
joints, feedback loop, reach check), inputs.csv (the 19 points of every step), first_frame.png, args.json;
session.csv per session. End a session with q (Ctrl+C can cut off the last video's conversion).
A dry run reads the motors only (torque untouched) and never moves anything.
"""
import argparse
import csv
import importlib.util
import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import ARM, HERE, JOINTS, URDF, calib_converter, matching_files
from fk_points import FKPoints
from liveview import LiveView

ENSEMBLE_W = [0.35, 0.25, 0.15, 0.11, 0.06, 0.03, 0.02, 0.01, 0.01, 0.01]  # paper's weights, newest plan first


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def demo_stats(data):
    """Start pose (mean first-frame joints) and joint range (+- 5 deg) of the training demos."""
    starts, qs = [], []
    for ep in sorted(Path(data).glob("ep_*")):
        st = list(csv.DictReader(open(ep / "robot_states.csv")))
        starts.append([float(st[0][f"follower_{j}"]) for j in JOINTS])
        qs += [[float(s[f"follower_{j}"]) for j in ARM] for s in st]
    qs = np.array(qs)
    return np.mean(starts, 0), qs.min(0) - 5.0, qs.max(0) + 5.0


class Overlay:
    """Draws everything on a frame (same picture for the live view and the video)."""

    def __init__(self, crop, mode):
        self.crop, self.mode, self.mask_contours = crop, mode, None

    def set_mask(self, mask):
        self.mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def draw(self, img, phase, obj=None, vis=None, ee=None, lines=()):
        out = img.copy()
        x0, y0, x1, y1 = self.crop
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 1)
        if self.mask_contours is not None:
            cv2.drawContours(out, self.mask_contours, -1, (0, 165, 255), 2)
        if obj is not None:
            for i, (x, y) in enumerate(obj):
                if not np.isnan(x):
                    seen = vis is None or bool(vis[i])
                    cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255) if seen else (255, 0, 0), -1)
        if ee is not None and not np.isnan(ee).any():
            cv2.polylines(out, [ee.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cv2.putText(out, "fk", tuple(ee[0].astype(int) + np.array([0, -8])), 0, 0.6, (0, 255, 0), 2)
        cv2.rectangle(out, (0, 0), (1280, 40 + 34 * len(lines)), (0, 0, 0), -1)
        cv2.putText(out, f"[{self.mode}] {phase}", (15, 30), 0, 0.9, (0, 255, 255), 2)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (15, 64 + 34 * i), 0, 0.8, (255, 255, 255), 2)
        return out


class Graphs:
    """Live strip charts under the camera image (live view and video), last `window` seconds:
    1. gripper tip x / y / z (cm): target (thin) vs the arm's real tip = FK of the measured joints (thick)
    2. gripper: P(closed) x 100 (newest plan and averaged), gripper command, measured jaw (0-100)
    3. feedback loop: largest joint error left after waiting (deg), wait (0.1 s units), largest sag offset (deg)"""

    H, PANELS = 300, 3

    def __init__(self, width=1280, window=20.0):
        self.w, self.win = width, window
        self.t, self.s = [], {}

    def add(self, t, **vals):
        self.t.append(t)
        for k, v in vals.items():
            self.s.setdefault(k, []).append(float(v))

    def _panel(self, title, series, lo=None, hi=None):
        pw, ph = self.w // self.PANELS, self.H
        img = np.full((ph, pw, 3), 22, np.uint8)
        cv2.putText(img, title, (8, 18), 0, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        t = np.array(self.t)
        if len(t) < 2:
            return img
        keep = t >= t[-1] - self.win
        t0 = max(t[-1] - self.win, 0.0)
        vals = [np.array(self.s[k])[keep] for k, *_ in series]
        allv = np.concatenate([v[np.isfinite(v)] for v in vals] or [np.zeros(1)])
        if allv.size == 0:  # nothing to plot yet (e.g. the feedback loop in a dry run)
            allv = np.zeros(1)
        lo = np.nanmin(allv) if lo is None else lo
        hi = np.nanmax(allv) if hi is None else hi
        if hi - lo < 1e-6:
            hi = lo + 1
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        x0, x1, y0, y1 = 44, pw - 6, 40, ph - 22
        for g in np.linspace(lo, hi, 5):
            y = int(y1 - (g - lo) / (hi - lo) * (y1 - y0))
            cv2.line(img, (x0, y), (x1, y), (55, 55, 55), 1)
            cv2.putText(img, f"{g:.0f}", (2, y + 4), 0, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, f"{t0:.0f} s", (x0, ph - 6), 0, 0.35, (150, 150, 150), 1)
        cv2.putText(img, f"{t0 + self.win:.0f} s", (x1 - 34, ph - 6), 0, 0.35, (150, 150, 150), 1)
        tx = x0 + (t[keep] - t0) / max(self.win, 1e-6) * (x1 - x0)
        n_leg = max(3, sum(1 for _, lb, *_ in series if lb))
        n_lab = 0
        for (key, label, color, thick), v in zip(series, vals):
            ok = np.isfinite(v)
            pts = np.stack([tx[ok], y1 - (v[ok] - lo) / (hi - lo) * (y1 - y0)], 1).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)
            if label:
                last = v[ok][-1] if ok.any() else float("nan")
                cv2.putText(img, f"{label} {last:.1f}", (8 + n_lab * (pw // n_leg), 34), 0, 0.42, color, 1,
                            cv2.LINE_AA)
                n_lab += 1
        return img

    def draw(self):
        X, Y, Z = (80, 80, 255), (80, 220, 80), (255, 140, 60)
        p1 = self._panel("tip cm: thick = real (FK), thin = target",
                         [("mx", "x", X, 2), ("my", "y", Y, 2), ("mz", "z", Z, 2),
                          ("tx", "", X, 1), ("ty", "", Y, 1), ("tz", "", Z, 1)])
        p2 = self._panel("gripper 0-100 (P% = P(closed) x 100)",
                         [("pc", "P%", (0, 165, 255), 2), ("pe", "P ens%", (0, 255, 0), 1),
                          ("gc", "cmd", (255, 255, 255), 1), ("gm", "jaw", (255, 255, 0), 2)], 0, 100)
        p3 = self._panel("feedback: leftover max joint err deg, wait x0.1 s",
                         [("le", "left err", (0, 255, 255), 2), ("wt", "wait", (255, 0, 255), 2),
                          ("sg", "|sag offs|", (255, 255, 255), 1)], 0, None)
        out = np.hstack([p1, p2, p3])
        return out if out.shape[1] == self.w else cv2.resize(out, (self.w, self.H))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=str(HERE / "models/mug40fk_gl_h"), help="models/<name>")
    p.add_argument("--ckpt", default="final_bc_model.pt")
    p.add_argument("--data", default=str(HERE / "data/mug_a2a"), help="training data (start pose, joint range)")
    p.add_argument("--reference", default=str(HERE / "references/mug_a2a"))
    p.add_argument("--execute", action="store_true", help="really move the arm (default: dry run)")
    p.add_argument("--session", action="store_true", help="many rollouts, controlled from this terminal")
    p.add_argument("--no_home", action="store_true", help="with --execute: don't move to the start pose first")
    p.add_argument("--seconds", type=float, default=60, help="longest rollout (the feedback loop makes it slow)")
    p.add_argument("--hz", type=float, default=10, help="at most this many policy steps per second")
    p.add_argument("--leash", type=float, default=10.0, help="target kept within this many deg of the measured joints")
    p.add_argument("--settle_tol", type=float, default=1.0, help="deg, every arm joint")
    p.add_argument("--settle_max", type=float, default=0.4, help="s, longest wait per step")
    p.add_argument("--sag_ki", type=float, default=0.3, help="gain of the sag offset (0 = off)")
    p.add_argument("--open", type=float, default=45.0, help="gripper command when open (demos: median 46)")
    p.add_argument("--closed", type=float, default=0.0)
    p.add_argument("--grip_on", type=float, default=0.6)
    p.add_argument("--grip_off", type=float, default=0.3)
    p.add_argument("--grip_min_closed", type=float, default=2.0)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--id", default="auto", help="follower calibration id; auto = the saved file that matches the motors")
    p.add_argument("--view_port", type=int, default=8770)
    p.add_argument("--gpu", default="1", help="CUDA device (default 1: GPU 0 of the lab PC is shared)")
    a = p.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    mode = "EXECUTE" if a.execute else "DRY RUN"
    view = LiveView(a.view_port)
    view.message(f"{mode}\nloading models ...")
    track = load_module("track", "3_track.py")
    train = load_module("train", "7_train.py")
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    import pyrealsense2 as rs

    # ---- policy, models, reference
    ck = torch.load(Path(a.policy) / a.ckpt, map_location="cuda", weights_only=False)
    cfg = ck["config"]
    assert cfg.get("ACTION") == "joints" and cfg.get("TOKEN_DIM") == 2, "this script runs joint-action (x, y) models"
    HIST = cfg.get("HISTORY", [0])
    policy = train.TinyTransformer(cfg["K"], cfg["ACTION_WINDOW"], cfg["D_MODEL"], cfg["NHEAD"], cfg["LAYERS"],
                                   cfg["D_FF"], action_dim=5, token_dim=2, n_hist=len(HIST)).cuda().eval()
    policy.load_state_dict(ck["model_state"])
    x_mean, x_std = ck["scaler_state"]
    a_mean, a_std = ck["act_scaler_state"]
    start_q, q_lo, q_hi = demo_stats(a.data)
    fkp = FKPoints()
    print(f"policy {a.policy}: joint changes, (x, y) tokens, history {HIST} steps back, gripper points from the joints")
    ref = json.loads((Path(a.reference) / "params.json").read_text())
    ref_img = cv2.imread(str(Path(a.reference) / "ref_image.png"))
    ref_mask = cv2.imread(str(Path(a.reference) / "ref_mask.png"), 0) > 0
    ooal, names, _ = track.load_ooal("cuda")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").cuda().eval()
    kin = RobotKinematics(str(URDF), "gripper_frame_link", ARM)
    ov = Overlay(ref["crop"], mode)

    # ---- camera (as recorded: RealSense 1280x720, 15 fps on the USB 2.1 link, auto exposure)
    pipe, rcfg = rs.pipeline(), rs.config()
    rcfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    pipe.start(rcfg)
    for _ in range(20):
        pipe.wait_for_frames()

    def grab():
        return np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()).copy()

    def grab_fresh():
        """newest frame: drop what is queued (the arm may have waited), then wait for the next one"""
        while pipe.poll_for_frames():
            pass
        return grab()

    def poll():
        fs = pipe.poll_for_frames()
        return np.asanyarray(fs.get_color_frame().get_data()).copy() if fs else None

    # ---- robot: angles in the training data's calibration, converted exactly to / from the motors' calibration
    data_calib = json.loads((sorted(Path(a.data).glob("ep_*"))[0] / "params.json").read_text())["calibration_used"]
    motor_id = a.id
    if motor_id == "auto":
        found = matching_files(SO101Follower(SO101FollowerConfig(port=a.port, id=data_calib)))
        if not found:
            sys.exit("no saved calibration matches the follower's motors (see README: arm calibration)")
        motor_id = data_calib if data_calib in found else found[0]
    print(f"follower calibration in the motors: {motor_id}; training data: {data_calib}")
    to_data, to_motor = calib_converter(motor_id, data_calib), calib_converter(data_calib, motor_id)
    robot = SO101Follower(SO101FollowerConfig(port=a.port, id=motor_id, max_relative_target=10.0))
    if a.execute:
        robot.connect(calibrate=False)  # torque on; never writes calibration
    else:
        robot.bus.connect()  # dry run: read only, motors left exactly as they are
    if not robot.bus.is_calibrated:
        sys.exit(f"follower doesn't match calibration {motor_id}")

    def read_q():
        o = robot.bus.sync_read("Present_Position")
        return np.array([to_data(j, o[j]) for j in JOINTS])

    def send(q):
        robot.send_action({f"{j}.pos": float(to_motor(j, v)) for j, v in zip(JOINTS, q)})

    # ---- live view + video
    st = {"graphs": Graphs(1280), "video": None, "raw": None, "t_video": [time.perf_counter()]}

    def out(frame, realtime=True):
        """Live view (+ video while a rollout is saved); realtime keeps the video at true speed."""
        frame = np.vstack([frame, st["graphs"].draw()])
        view.show(frame)
        if st["video"] is None:
            return
        if not realtime:
            st["video"].write(frame)
            return
        now = time.perf_counter()
        while st["t_video"][0] <= now:
            st["video"].write(frame)
            st["t_video"][0] += 1 / a.hz

    def new_run(run_dir):
        run_dir.mkdir(parents=True)
        (run_dir / "args.json").write_text(json.dumps(vars(a), indent=2))
        st["graphs"] = Graphs(1280)
        st["raw"] = run_dir / "video_raw.mp4"
        st["video"] = cv2.VideoWriter(str(st["raw"]), cv2.VideoWriter_fourcc(*"mp4v"), a.hz, (1280, 720 + Graphs.H))
        st["t_video"][0] = time.perf_counter()

    def finish_video(run_dir):
        if st["video"] is None:
            return
        st["video"].release()
        st["video"] = None
        ff = shutil.which("ffmpeg") or str(Path(sys.executable).resolve().parents[3] / "bin/ffmpeg")
        if Path(ff).exists():
            r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(st["raw"]), "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", "23", str(run_dir / "video.mp4")])
            if r.returncode == 0:
                st["raw"].unlink()
        else:
            print("ffmpeg not found: kept video_raw.mp4 (mp4v)")
        print(f"saved -> {run_dir}  (video.mp4)")

    def move_slowly(goal, phase, deg_per_step=1.0):
        q0 = read_q()
        n = max(1, int(np.abs(goal - q0).max() / deg_per_step))
        for i, q in enumerate(np.linspace(q0, goal, n + 1)[1:]):
            send(q)
            img = poll()
            if img is not None:
                out(ov.draw(img, f"{phase} {i + 1}/{n}", lines=[f"goal {np.round(goal, 1)}"]))
            time.sleep(1 / 30)

    class Keys:
        """Lines typed in the terminal, read without blocking (os.read on stdin, so no line hides in a buffer)."""

        def __init__(self):
            self.fd, self.buf, self.lines = sys.stdin.fileno(), "", []

        def get(self, timeout=0.0):
            if not self.lines:
                r, _, _ = select.select([self.fd], [], [], timeout)
                if r:
                    data = os.read(self.fd, 1024).decode(errors="ignore")
                    if data == "":  # stdin closed
                        self.lines.append("q")
                    self.buf += data
                    while "\n" in self.buf:
                        line, self.buf = self.buf.split("\n", 1)
                        self.lines.append(line.strip().lower())
            return self.lines.pop(0) if self.lines else None

    keys = Keys()

    def wait_key(phase, lines, still=None):
        """live camera (or `still`, an already drawn picture) in the live view until a line is typed; returns it"""
        print("   " + "   ".join(lines), flush=True)
        while True:
            if still is not None:
                out(still)
            else:
                img = poll()
                if img is not None:
                    out(ov.draw(img, phase, lines=lines))
            k_ = keys.get(0.05)
            if k_ is not None:
                return k_

    def replay(run_dir):
        cap = cv2.VideoCapture(str(run_dir / "video.mp4" if (run_dir / "video.mp4").exists() else st["raw"]))
        while True:
            ok, fr = cap.read()
            if not ok or keys.get() is not None:
                break
            view.show(fr)
            time.sleep(1 / a.hz)
        cap.release()

    def first_frame():
        """OOAL + DINO on the current frame, as in step 3 -> (image, points, info line)"""
        img0 = grab_fresh()
        out(ov.draw(img0, "OOAL + DINO on the first frame ..."))
        mask, _, peaks = track.affordance_mask(img0, ooal, names, ref["affs"], ref["mask_thresh"], ref["dilate"],
                                               ref["crop"])
        box = track.roi(mask, ref["pad"], img0.shape[:2])
        ov.set_mask(mask)
        pts, n_pairs, n_both = track.object_points(ref_img, ref_mask, tuple(ref["roi_y0y1x0x1"]), img0, mask, box,
                                                   argparse.Namespace(**ref))
        info = (f"OOAL '{ref['affs'][0]}' peak {peaks[ref['affs'][0]]:.2f}, kept 1 of {peaks['blobs']} blobs; "
                f"DINO {n_pairs} pairs, {n_both} in both masks -> {len(pts)} points")
        print(info)
        return img0, pts, info

    def rollout(run_dir, img0, pts, info, stop_on_enter=False):
        """one rollout from the first frame (its handle points already picked). Returns (seconds, stopped early)."""
        graphs, t_video = st["graphs"], st["t_video"]
        stopped = False
        first = ov.draw(img0, "first frame", obj=pts, lines=[info])
        cv2.imwrite(str(run_dir / "first_frame.png"), first)
        for _ in range(int(1.5 * a.hz)):  # hold the first-frame picture 1.5 s in the view and the video
            out(first, realtime=False)
        if a.execute:  # the demos always open the gripper first: give the policy its "go" state from step 1
            goal = read_q()
            goal[5] = a.open
            move_slowly(goal, "OPENING GRIPPER", deg_per_step=1.5)
            time.sleep(0.3)
        n_pts = len(pts)
        frames_rgb = []
        queries = torch.cat([torch.zeros(n_pts, 1), torch.tensor(pts, dtype=torch.float32)], 1)[None].cuda()
        obj = np.full((15, 2), np.nan)
        obj[:n_pts] = pts
        vis = np.ones(15, bool)
        step = cotracker.step
        t_points = 0.0
        q_cmd = read_q().copy()
        grip_closed, t_closed = (q_cmd[5] < 20), -1e9
        tok_hist = {}  # policy step -> its 19 tokens (history input)
        plans = {}  # policy step -> (10 x 5 joint changes, 10 x P(closed))
        sag = np.zeros(5)  # feedback loop: offset added to the commanded arm joints
        settle_ms, reached, left_err = np.nan, np.nan, np.nan
        q_sent_prev, q_tgt_prev = None, None
        log = open(run_dir / "log.csv", "w", newline="")
        lw = csv.writer(log)
        lw.writerow(["t", "frame", "steps_applied", "late_ms", "p_closed", "p_ens", "grip_closed"]
                    + [f"d_{j}" for j in ARM] + ["tgt_x", "tgt_y", "tgt_z"] + [f"cmd_{j}" for j in JOINTS]
                    + [f"meas_{j}" for j in JOINTS] + ["meas_tip_x", "meas_tip_y", "meas_tip_z"]
                    + ["settle_ms", "reached", "left_err_deg"] + [f"sag_{j}" for j in ARM]
                    + ["reach_tip_mm", "reach_rot_deg", "reach_joint_max_deg"] + [f"err_{j}" for j in ARM])
        inp = open(run_dir / "inputs.csv", "w", newline="")
        iw = csv.writer(inp)
        iw.writerow(["t", "frame"] + [f"g{i}_{c}" for i in range(1, 5) for c in "xy"]
                    + [f"h{i}_{c}" for i in range(1, 16) for c in "xy"] + [f"vis{i}" for i in range(1, 16)])
        print(f"{mode} for at most {a.seconds:.0f} s. "
              + ("Enter = stop early, Ctrl+C = end the session." if stop_on_enter else "Ctrl+C stops."))
        t_start = time.perf_counter()
        t_video[0] = t_start
        k, frame_no = 0, 0
        try:
            while True:
                t_loop = time.perf_counter()
                elapsed = t_loop - t_start
                if elapsed > a.seconds:
                    break
                if stop_on_enter and keys.get() is not None:
                    stopped = True
                    print("[rollout] stopped early")
                    break
                frame_no += 1  # one policy step per loop: the arm sets the pace
                img = grab_fresh() if k > 0 else img0
                frames_rgb.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                q_meas = read_q()
                ee = fkp.box(q_meas[:5], q_meas[5])  # 4 gripper points from the joints read with this image

                # handle points: CoTracker online, 8-frame steps on a 16-frame window
                with torch.no_grad():
                    if k == 2 * step - 1:
                        vid = torch.stack(frames_rgb[:2 * step]).permute(0, 3, 1, 2)[None].float().cuda()
                        cotracker(video_chunk=vid, is_first_step=True, queries=queries)
                    if k >= 2 * step - 1 and (k + 1) % step == 0:
                        ind = k + 1 - 2 * step
                        vid = torch.stack(frames_rgb[ind:ind + 2 * step]).permute(0, 3, 1, 2)[None].float().cuda()
                        tr, vv = cotracker(video_chunk=vid)
                        obj[:n_pts], vis[:n_pts] = tr[0, -1].cpu().numpy(), vv[0, -1].cpu().numpy() > 0
                        t_points = elapsed
                        frames_rgb[:ind + step] = [None] * (ind + step)  # free old frames (keep indices)

                # tokens -> policy (history: the newest step <= now - lag; the first step stands in before the start)
                tok = np.concatenate([ee, obj])
                iw.writerow([f"{elapsed:.3f}", frame_no] + [f"{v:.2f}" for v in tok.ravel()] + [int(v) for v in vis])
                tok_hist[frame_no] = tok.copy()
                past = sorted(tok_hist)
                x_tok = np.concatenate([tok_hist[max([f for f in past if f <= frame_no - L], default=past[0])]
                                        for L in HIST], axis=0)
                for f in [f for f in past if f < frame_no - max(HIST) - 5]:
                    tok_hist.pop(f)
                m = np.isnan(x_tok).any(1)
                x = np.nan_to_num((x_tok - x_mean) / x_std)
                with torch.no_grad():
                    pr, pg = policy(torch.tensor(x, dtype=torch.float32)[None].cuda(), torch.tensor(m)[None].cuda())
                plans[frame_no] = (pr[0].cpu().numpy().reshape(-1, 5) * a_std + a_mean, pg[0].cpu().numpy())

                # reach check: did the arm get to the joints sent last step? (joints, FK tip position / orientation)
                q_meas = read_q()  # fresh reading for control
                F = kin.forward_kinematics(q_meas[:5])
                if q_sent_prev is not None:
                    jerr = q_sent_prev - q_meas[:5]
                    Fs = kin.forward_kinematics(q_sent_prev)
                    r_tip = 1000 * np.linalg.norm(Fs[:3, 3] - F[:3, 3])
                    r_rot = np.degrees(np.arccos(np.clip((np.trace(Fs[:3, :3].T @ F[:3, :3]) - 1) / 2, -1.0, 1.0)))
                else:
                    jerr, r_tip, r_rot = np.full(5, np.nan), np.nan, np.nan

                # temporal ensembling (plans made i steps ago, their step i) -> joint change and P(closed)
                num, num_p, den = np.zeros(5), 0.0, 0.0
                for i, w in enumerate(ENSEMBLE_W):
                    if frame_no - i in plans:
                        num += w * plans[frame_no - i][0][i]
                        num_p += w * float(plans[frame_no - i][1][i])
                        den += w
                d = num / den
                p_closed, p_ens = float(plans[frame_no][1][0]), num_p / den
                for f in [f for f in plans if f <= frame_no - len(ENSEMBLE_W)]:
                    plans.pop(f)

                # joint target = previous target + change, near the arm and inside the demos' range
                q_tgt = (q_meas[:5].copy() if q_tgt_prev is None else q_tgt_prev.copy()) + d
                q_tgt = np.clip(np.clip(q_tgt, q_meas[:5] - a.leash, q_meas[:5] + a.leash), q_lo, q_hi)
                q_tgt_prev = q_tgt.copy()
                q_cmd[:5] = q_tgt
                target_p = kin.forward_kinematics(q_tgt)[:3, 3]

                # gripper: averaged P with hysteresis, stays closed >= grip_min_closed s
                if not grip_closed and p_ens > a.grip_on:
                    grip_closed, t_closed = True, elapsed
                elif grip_closed and p_ens < a.grip_off and elapsed - t_closed >= a.grip_min_closed:
                    grip_closed = False
                q_cmd[5] = a.closed if grip_closed else a.open

                if a.execute:  # feedback loop: send (target + sag offset), read until reached or settle_max
                    q_send = q_cmd.copy()
                    q_send[:5] = q_cmd[:5] + sag
                    t_s = time.perf_counter()
                    while True:
                        send(q_send)
                        time.sleep(0.02)
                        err = q_cmd[:5] - read_q()[:5]
                        if np.abs(err).max() < a.settle_tol or time.perf_counter() - t_s > a.settle_max:
                            break
                    settle_ms = 1000 * (time.perf_counter() - t_s)
                    reached = float(np.abs(err).max() < a.settle_tol)
                    left_err = float(np.abs(err).max())
                    small = np.abs(err) < 3.0  # only near the target: sag, not a move still under way
                    sag = np.clip(sag + a.sag_ki * err * small, -3.0, 3.0)
                    q_sent_prev = q_cmd[:5].copy()

                tip = F[:3, 3]
                jmax = np.nanmax(np.abs(jerr)) if np.isfinite(jerr).any() else np.nan
                graphs.add(elapsed, mx=100 * tip[0], my=100 * tip[1], mz=100 * tip[2], tx=100 * target_p[0],
                           ty=100 * target_p[1], tz=100 * target_p[2], pc=100 * p_closed, pe=100 * p_ens,
                           gc=q_cmd[5], gm=q_meas[5], le=left_err, wt=settle_ms / 100, sg=np.abs(sag).max())
                late = (time.perf_counter() - t_loop) * 1000 - 1000 / a.hz
                lw.writerow([f"{elapsed:.3f}", frame_no, 1, f"{max(0, late):.0f}", f"{p_closed:.3f}", f"{p_ens:.3f}",
                             int(grip_closed), *[f"{v:.5f}" for v in d], *[f"{v:.4f}" for v in target_p],
                             *[f"{v:.2f}" for v in q_cmd], *[f"{v:.2f}" for v in q_meas], *[f"{v:.4f}" for v in tip],
                             f"{settle_ms:.0f}", f"{reached:.0f}", f"{left_err:.2f}", *[f"{v:.2f}" for v in sag],
                             f"{r_tip:.1f}", f"{r_rot:.2f}", f"{jmax:.2f}", *[f"{v:.2f}" for v in jerr]])
                out(ov.draw(img, f"RUNNING  t {elapsed:5.1f} s  step {frame_no}", obj=obj[:n_pts], vis=vis[:n_pts],
                            ee=ee, lines=[
                                f"P(closed) {p_closed:.2f} (ens {p_ens:.2f}) -> gripper "
                                f"{'CLOSED' if grip_closed else 'open'}   points updated {elapsed - t_points:.1f} s ago"
                                + (f"   reached {'yes' if reached == 1 else 'NO'} in {settle_ms:.0f} ms, left "
                                   f"{left_err:.1f} deg" if a.execute else ""),
                                f"target tip {np.round(100 * target_p, 1)} cm   joint step {np.round(d, 2)} deg",
                                f"reach check (sent last step vs now): tip {r_tip:.1f} mm, rot {r_rot:.1f} deg, "
                                f"joints max {jmax:.1f} deg"]))
                if k % 10 == 0:
                    print(f"t {elapsed:5.1f}s  P(closed) {p_ens:.2f}  gripper {'closed' if grip_closed else 'open'}  "
                          f"target tip cm {np.round(100 * target_p, 1)}", flush=True)
                k += 1
                time.sleep(max(0.0, 1 / a.hz - (time.perf_counter() - t_loop)))
        finally:
            log.close()
            inp.close()
        return time.perf_counter() - t_start, stopped

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = None
    try:
        q_now = read_q()
        print(f"start pose of the demos: {np.round(start_q, 1)}; follower now: {np.round(q_now, 1)}")
        if a.execute and not a.no_home:
            print("going home (demos' start pose), slowly ...")
            move_slowly(start_q, "HOMING")
            time.sleep(0.5)
        elif np.abs(q_now[:5] - start_q[:5]).max() > 15:
            print("WARNING: the follower is > 15 deg from the demos' start pose")

        if not a.session:  # ---- one rollout
            run_dir = HERE / "runs" / f"{Path(a.policy).name}_{stamp}{'' if a.execute else '_dry'}"
            new_run(run_dir)
            img0, pts, info = first_frame()
            if len(pts) == 0:
                sys.exit("no handle points found: is the mug inside the box?")
            rollout(run_dir, img0, pts, info)
        else:  # ---- session: many rollouts, controlled from this terminal
            sess = HERE / "runs" / f"{Path(a.policy).name}_session_{stamp}{'' if a.execute else '_dry'}"
            sess.mkdir(parents=True)
            (sess / "args.json").write_text(json.dumps(vars(a), indent=2))
            sess_csv = sess / "session.csv"
            with open(sess_csv, "w", newline="") as f_:
                csv.writer(f_).writerow(["roll", "time", "points", "seconds", "stopped_early", "kept", "folder"])

            def note(row):
                with open(sess_csv, "a", newline="") as f_:
                    csv.writer(f_).writerow(row)

            print(f"\n[session] {sess.relative_to(HERE)}   live view: http://localhost:{a.view_port}")
            n = 0
            while True:
                name = f"roll_{n:03d}"
                print(f"\n[session] READY {name}: place the mug (arm holds the start pose)")
                key = wait_key(f"READY {name}: place the mug", ["Enter = check the first frame", "q = quit"])
                if key == "q":
                    break
                while True:  # CHECK: the handle points of this mug placement
                    img0, pts, info = first_frame()
                    shown = ov.draw(img0, f"CHECK {name}: {len(pts)} handle points (training: 15)", obj=pts,
                                    lines=[info, "Enter = run   c = check again (after moving the mug)   q = quit"])
                    print(f"[session] CHECK {name}: {len(pts)} handle points (training: always 15)")
                    key = wait_key(f"CHECK {name}", ["Enter = run", "c = check again (after moving the mug)",
                                                      "q = quit"], still=shown)
                    if key != "c":
                        break
                if key == "q":
                    break
                if len(pts) == 0:
                    print("[session] no handle points: place the mug inside the box and check again")
                    continue
                run_dir = sess / name
                new_run(run_dir)
                try:
                    secs, stopped = rollout(run_dir, img0, pts, info, stop_on_enter=True)
                except KeyboardInterrupt:
                    note([name, time.strftime("%H:%M:%S"), len(pts), "", "ctrl+c", "kept", name])
                    raise
                if a.execute:
                    print("[session] returning slowly to the start pose ...")
                    move_slowly(start_q, "RETURNING")
                finish_video(run_dir)
                while True:  # REVIEW
                    print(f"\n[session] REVIEW {name}")
                    key = wait_key(f"REVIEW {name}", ["Enter = keep", "d = discard", "r = replay"])
                    if key == "r":
                        print("[session] replaying in the live view (Enter stops)")
                        replay(run_dir)
                        continue
                    break
                kept = key != "d"
                folder = name
                if not kept:
                    dst = sess / "_discarded" / f"{name}_{time.strftime('%H%M%S')}"
                    dst.parent.mkdir(exist_ok=True)
                    shutil.move(str(run_dir), str(dst))
                    folder = str(dst.relative_to(sess))
                    print(f"[session] discarded -> {folder}")
                else:
                    print(f"[session] kept {name}")
                note([name, time.strftime("%H:%M:%S"), len(pts), f"{secs:.1f}", int(stopped),
                      "kept" if kept else "discarded", folder])
                run_dir = None
                n += 1
            print(f"[session] done: {sess_csv.relative_to(HERE)}")
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        try:
            if a.execute and robot.is_connected:
                print("returning slowly to the start pose ...")
                move_slowly(start_q, "RETURNING")
        except Exception as e:  # noqa: BLE001
            print("return failed:", e)
        if a.execute and robot.is_connected:
            print("disconnecting: the follower goes limp now, support it.")
            robot.disconnect()
        elif robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
        pipe.stop()
        if run_dir is not None:
            finish_video(run_dir)
        view.message(f"{mode} finished")


if __name__ == "__main__":
    main()
