"""Recording dashboard in the browser (localhost): the live camera, and the last saved demo with its quick checks and a
replay, to decide on the spot whether to keep it or redo it. Port of a2a_so101/a2a/liveview.py (the dashboard the
mug_a2a demos were recorded with), used by 1d_record.py.

Only one program can use the RealSense, so the recorder serves the view: update(frame, status) with each new frame,
set_last(folder, checks) after saving a demo. Over VS Code Remote-SSH the port is forwarded (PORTS tab).
"""
import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Recording</title>
<style>body{margin:0;padding:12px;background:#111;color:#ddd;font:14px system-ui,sans-serif}
#wrap{display:grid;grid-template-columns:1fr 1fr;gap:14px} img{max-width:100%;display:block;background:#000}
h3{margin:4px 0 8px;font-size:15px;color:#fff} ul{margin:6px 0;padding-left:18px} li{margin:2px 0}
.ok{color:#8f8} .bad{color:#fb5} button{padding:5px 12px;margin-right:6px} .note{color:#999;font-size:12px}
@media (max-width:1000px){#wrap{grid-template-columns:1fr}}</style></head><body>
<div id="wrap">
 <div><h3>Live camera</h3><img src="/stream" alt="live camera">
  <p class="note">Keep your hands and the leader arm out of the picture. Green outline = the ArUco marker on the
  gripper. Top right: the wrist camera.</p></div>
 <div><h3>Last saved demo: <span id="name">none yet</span></h3>
  <ul id="checks"></ul>
  <button id="p1">Replay</button><button id="p2">Replay 2x</button>
  <span class="note">Bad? Type r + Enter in the terminal to discard it (moved to _discarded/) and record it again.</span>
  <img id="rep" alt="" style="margin-top:8px"></div>
</div>
<script>
const $ = id => document.getElementById(id); let last = null;
async function refresh() {
  try {
    const d = await (await fetch('/last.json')).json();
    if (d.name === last) return;
    last = d.name; $('name').textContent = d.name || 'none yet';
    $('checks').innerHTML = (d.checks || []).map(([ok, s]) => `<li class="${ok ? 'ok' : 'bad'}">${ok ? '&#10003;' : '&#9888;'} ${s}</li>`).join('');
    $('rep').removeAttribute('src');
  } catch (e) {}
}
const play = s => { if (last) $('rep').src = `/replay.mjpg?speed=${s}&k=${Date.now()}`; };
$('p1').onclick = () => play(1); $('p2').onclick = () => play(2);
setInterval(refresh, 1500); refresh();
</script></body></html>"""


def _part(jpg):
    return b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"


def _text(img, s, org, color, scale=0.7):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _inset(img, small, w, corner="top"):
    """Paste `small` (the wrist camera; may be None) at the top right or bottom right of img, w px wide, white border."""
    if small is None:
        return
    h = int(small.shape[0] * w / small.shape[1])
    if corner == "bottom":
        H = img.shape[0] - 10  # above the progress bar
        img[H - h - 2:H, img.shape[1] - w - 2:] = 255
        img[H - h:H, img.shape[1] - w:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)
        return
    img[0:h + 2, img.shape[1] - w - 2:] = 255
    img[0:h, img.shape[1] - w:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)


def side_px(corners):
    c = np.asarray(corners)
    return float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))


def draw_markers(img, found, px_scale=1.0):
    """Outline, corner 0 (red dot), id and side length (px in the camera image) of each (id, corners) in found."""
    for mid, c in found:
        cv2.polylines(img, [np.round(c).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in c[0]), 6, (0, 0, 255), -1, cv2.LINE_AA)
        x, y = int(c[:, 0].max()) + 8, int(c[:, 1].min())
        for col, th in (((0, 0, 0), 4), ((0, 255, 0), 2)):
            cv2.putText(img, f"id {mid}  {side_px(c) / px_scale:.0f} px", (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        col, th, cv2.LINE_AA)
    return img


def read_states(ep):
    with open(Path(ep) / "states.csv") as f:
        rows = list(csv.DictReader(f))
    return {"frames": np.array([int(r["frame"]) for r in rows]), "t": np.array([float(r["t"]) for r in rows]),
            "follower": np.array([[float(r[f"follower_{j}"]) for j in JOINTS] for r in rows]),
            "command": np.array([[float(r[f"command_{j}"]) for j in JOINTS] for r in rows])}


def read_marker(ep):
    """-> dict frame -> (4, 2) corners or None; {} if the demo has no marker.csv."""
    path = Path(ep) / "marker.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {int(r["frame"]): (np.array([[float(r[f"c{k}_x"]), float(r[f"c{k}_y"])] for k in range(4)])
                                  if r["found"] == "1" else None) for r in csv.DictReader(f)}


class RecordView:
    def __init__(self, port, width=960, grip_thresh=5.0, wrist_cam=None, marker_det=None):
        self.width, self.grip_thresh, self.wrist_cam, self.marker_det = width, grip_thresh, wrist_cam, marker_det
        self.jpeg, self.cond, self.server = None, threading.Condition(), None
        self.last, self.checks = None, []
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _head(self, ctype, length=None):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()

            def do_GET(self):
                url = urlparse(self.path)
                try:
                    if url.path == "/":
                        self._head("text/html; charset=utf-8", len(PAGE))
                        self.wfile.write(PAGE)
                    elif url.path == "/stream":
                        self._head("multipart/x-mixed-replace; boundary=frame")
                        while True:
                            with view.cond:
                                view.cond.wait(timeout=2.0)
                                jpg = view.jpeg
                            if jpg is not None:
                                self.wfile.write(_part(jpg))
                    elif url.path == "/last.json":
                        body = json.dumps({"name": view.last.name if view.last else None,
                                           "checks": view.checks}).encode()
                        self._head("application/json", len(body))
                        self.wfile.write(body)
                    elif url.path == "/replay.mjpg" and view.last is not None:
                        speed = float(parse_qs(url.query).get("speed", ["1"])[0])
                        self._head("multipart/x-mixed-replace; boundary=frame")
                        view._replay(view.last, speed, self.wfile)
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as e:
            print(f"[view] port {port} is busy ({e}): no live view (pick another with --view_port)")
            return
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"[view] dashboard: http://localhost:{port}  (live camera + replay of the last demo; "
              "VS Code forwards the port: PORTS tab)")

    def update(self, frame, status="", color=(255, 255, 255), wrist=None, markers=None):
        """Show this frame with the marker, a status line and the wrist camera as an inset at the top right."""
        if self.server is None or frame is None:
            return
        img = frame.copy()
        if self.marker_det is not None:  # markers = [(id, corners)] already found by the caller, else detect here
            found = self.marker_det.detect_all(frame) if markers is None else markers
            draw_markers(img, found)
            status = (status + "   " if status else "") + ("MARKER OK" if found else "MARKER NOT SEEN")
        img = cv2.resize(img, (self.width, int(img.shape[0] * self.width / img.shape[1])), interpolation=cv2.INTER_AREA)
        if wrist is None and self.wrist_cam is not None:
            wrist = self.wrist_cam.peek()
        _inset(img, wrist, self.width // 4)
        if status:
            _text(img, status, (12, 32), color, 0.8)
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        with self.cond:
            self.jpeg = jpg
            self.cond.notify_all()

    def set_last(self, folder, checks):
        """The demo just saved (shown with its checks and the replay buttons); None after it was discarded."""
        self.last, self.checks = (Path(folder) if folder else None), [[bool(ok), s] for ok, s in checks]

    def _replay(self, folder, speed, out):
        """Stream a saved demo's frames at its recorded pace, with time, gripper and a progress bar."""
        st = read_states(folder)
        t, cmd, fol, frames = st["t"], st["command"], st["follower"], st["frames"]
        mk = read_marker(folder)
        w = 640
        t_play = time.perf_counter()
        for i, fid in enumerate(frames):
            img = cv2.imread(str(folder / f"{fid}.png"))
            if img is None:
                continue
            scale = w / img.shape[1]
            img = cv2.resize(img, (w, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            c = mk.get(int(fid))
            if c is not None:
                draw_markers(img, [(0, c * scale)], px_scale=scale)
            _inset(img, cv2.imread(str(folder / "wrist" / f"{fid}.jpg")), w // 3, corner="bottom")
            g = cmd[i, 5]
            _text(img, f"REPLAY {folder.name}  t={t[i]:4.1f}/{t[-1]:.1f} s", (10, 26), (255, 255, 255), 0.65)
            _text(img, f"gripper {g:3.0f} {'CLOSED' if g < self.grip_thresh else 'open'}  jaw {fol[i, 5]:3.0f}",
                  (10, 54), (0, 200, 255) if g < self.grip_thresh else (120, 255, 120), 0.65)
            if mk:
                _text(img, "marker seen" if c is not None else "marker NOT seen", (10, 82),
                      (0, 255, 0) if c is not None else (0, 0, 255), 0.65)
            h = img.shape[0]
            cv2.rectangle(img, (0, h - 8), (int(w * (i + 1) / len(frames)), h), (0, 200, 255), -1)
            out.write(_part(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()))
            wait = t_play + t[i] / speed - time.perf_counter()  # keep the recorded pace
            if wait > 0:
                time.sleep(wait)

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
