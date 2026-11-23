"""Live view in the browser (works over SSH: VS Code forwards the port; open http://localhost:<port>).

    view = LiveView(8770)   # starts a background web server
    view.show(img_bgr)      # newest frame; the page shows it as a stream (MJPEG)
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = b"""<!doctype html><html><head><title>a2a_paper live</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif}img{width:100%;max-width:1280px;display:block;margin:auto}</style>
</head><body><img src="/stream"></body></html>"""


class LiveView:
    def __init__(self, port=8770, quality=80):
        self.jpg, self.cond, self.quality = None, threading.Condition(), quality
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while True:
                        with view.cond:
                            view.cond.wait_for(lambda: view.jpg is not last, timeout=1.0)
                            jpg = last = view.jpg
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"live view: http://localhost:{port}  (VS Code forwards the port)")

    def show(self, img_bgr):
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.cond:
                self.jpg = buf.tobytes()
                self.cond.notify_all()

    def message(self, text, size=(1280, 720)):
        """A plain status screen (e.g. while models load)."""
        import numpy as np
        img = np.zeros((size[1], size[0], 3), np.uint8)
        for i, line in enumerate(text.split("\n")):
            cv2.putText(img, line, (40, 80 + 50 * i), 0, 1.2, (255, 255, 255), 2)
        self.show(img)

    def close(self):
        self.server.shutdown()
        time.sleep(0.1)
