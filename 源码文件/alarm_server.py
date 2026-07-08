#!/usr/bin/env python3
# 4x4 matrix pressure alarm + HTTP service (port 8001, GET /alarm)

import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import gpiod

# BOARD physical pin mapping
COLS = [37, 36, 31, 29]    # input + internal pull-up
ROWS = [22, 18, 16, 15]    # output (active-low scan)
M = {37: 401, 36: 381, 31: 400, 29: 399, 22: 387, 18: 402, 16: 382, 15: 388}
CHIP = gpiod.Chip("gpiochip4")

# 4x4 alarm matrix, True means active
alarm = [[False] * 4 for _ in range(4)]

# GPIO init
col_lines, row_lines = [], []
for p in COLS:
    ln = CHIP.get_line(M[p] - 379)
    ln.request(
        consumer="kp_col",
        type=gpiod.LINE_REQ_DIR_IN,
        flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP,
    )
    col_lines.append(ln)
for p in ROWS:
    ln = CHIP.get_line(M[p] - 379)
    ln.request(consumer="kp_row", type=gpiod.LINE_REQ_DIR_OUT, default_val=1)
    row_lines.append(ln)
for row in row_lines:
    row.set_value(1)
time.sleep(0.05)

# Scan params:
# Keep debounce at 3 to resist motor noise, but scan faster so short presses
# still trigger more easily than before.
DEBOUNCE = int(os.environ.get("FY_PRESS_DEBOUNCE", "3"))
SETTLE_US = 2000
SCAN_GAP_S = float(os.environ.get("FY_PRESS_SCAN_GAP_S", "0.001"))
press_cnt = [[0] * 4 for _ in range(4)]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/alarm" or self.path.startswith("/alarm?"):
            grid = [[1 if alarm[r][c] else 0 for c in range(4)] for r in range(4)]
            body = (
                '{"grid":' + str(grid) +
                ',"rows":[' + ",".join(str(r) for r in ROWS) + ']' +
                ',"cols":[' + ",".join(str(c) for c in COLS) + ']}'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"4x4 alarm server ok\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


def cleanup(*_):
    for ln in row_lines + col_lines:
        try:
            ln.release()
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def http_thread():
    HTTPServer(("0.0.0.0", 8001), Handler).serve_forever()


import threading

threading.Thread(target=http_thread, daemon=True).start()
print(
    f"[alarm_server] HTTP on :8001 GET /alarm GET /health "
    f"debounce={DEBOUNCE} scan_gap_s={SCAN_GAP_S}",
    flush=True,
)


try:
    while True:
        for ri, _ in enumerate(row_lines):
            for rj, row in enumerate(row_lines):
                row.set_value(0 if rj == ri else 1)
            time.sleep(SETTLE_US / 1e6)
            for ci, col in enumerate(col_lines):
                if col.get_value() == 0:
                    press_cnt[ri][ci] += 1
                else:
                    press_cnt[ri][ci] = 0

        for r in range(4):
            for c in range(4):
                if press_cnt[r][c] >= DEBOUNCE and not alarm[r][c]:
                    alarm[r][c] = True
                    print(f"[ALARM ON ] r={r} c={c}", flush=True)
                elif press_cnt[r][c] == 0 and alarm[r][c]:
                    alarm[r][c] = False
                    print(f"[ALARM OFF] r={r} c={c}", flush=True)
        time.sleep(SCAN_GAP_S)
except KeyboardInterrupt:
    cleanup()
