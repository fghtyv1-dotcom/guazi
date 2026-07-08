#!/usr/bin/env python3
"""Offline video inference with optional relay/alarm dispatch."""

import argparse
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn


MODEL = "/home/sunrise/test/best_aivideo.bin"
VIDEO = "/home/sunrise/test/测试视频2.mp4"
OUTPUT = "/home/sunrise/test/result_测试视频2.mp4"
T = 640
CONF = 0.25
NMS_I = 0.45
CLASSES = ["crested_ibis", "hunter", "other_birds", "weasel", "snake"]
ALARM_SPECIES = {"hunter", "weasel", "snake"}
DEFAULT_ALARM_HOLD_S = 5.0


def bgr2nv12(bgr):
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape((h * w * 3 // 2,))
    a = h * w
    uv = yuv[a:].reshape((2, a // 4)).transpose((1, 0)).reshape((a // 2,))
    out = np.zeros_like(yuv)
    out[:a] = yuv[:a]
    out[a:] = uv
    return out


def letterbox(img, new=T):
    h0, w0 = img.shape[:2]
    s = min(new / h0, new / w0)
    nh, nw = int(h0 * s), int(w0 * s)
    pad = np.full((new, new, 3), 114, dtype=np.uint8)
    dy = (new - nh) // 2
    dx = (new - nw) // 2
    pad[dy : dy + nh, dx : dx + nw] = cv2.resize(img, (nw, nh))
    return pad, s, dx, dy


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes, scores, iou_thr):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        iw = np.maximum(0, xx2 - xx1)
        ih = np.maximum(0, yy2 - yy1)
        inter = iw * ih
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        idxs = idxs[1:][inter / (a1 + a2 - inter + 1e-6) <= iou_thr]
    return keep


def infer_frame(model, frame, h0, w0):
    pad, s, dx, dy = letterbox(frame, T)
    nv12 = bgr2nv12(pad)
    outs = model[0].forward(nv12)
    strides = [8, 16, 32]
    boxes_l, scores_l, clses_l = [], [], []
    for i, st in enumerate(strides):
        cls = outs[2 * i].buffer[0]
        box = outs[2 * i + 1].buffer[0]
        h, w = cls.shape[:2]
        p = sigmoid(cls)
        pmax = p.max(axis=-1)
        cid = p.argmax(axis=-1)
        mask = pmax >= CONF
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        conf = pmax[ys, xs]
        cidp = cid[ys, xs]
        cx = (sigmoid(box[ys, xs, 0]) + xs) * st
        cy = (sigmoid(box[ys, xs, 1]) + ys) * st
        box_scale = 1.8
        bw = (2 * sigmoid(box[ys, xs, 2])) ** 2 * st * box_scale
        bh = (2 * sigmoid(box[ys, xs, 3])) ** 2 * st * box_scale
        boxes_l.append(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1))
        scores_l.append(conf)
        clses_l.append(cidp)
    if not boxes_l:
        return frame, []
    boxes = np.concatenate(boxes_l)
    scores = np.concatenate(scores_l)
    clses = np.concatenate(clses_l)
    keep = nms(boxes, scores, NMS_I)
    boxes, scores, clses = boxes[keep], scores[keep], clses[keep]
    boxes[:, [0, 2]] -= dx
    boxes[:, [1, 3]] -= dy
    boxes /= s
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)
    for (x1, y1, x2, y2), sc, ci in zip(boxes, scores, clses):
        name = CLASSES[int(ci)] if int(ci) < len(CLASSES) else f"cls_{ci}"
        color = (0, 0, 255) if name in ALARM_SPECIES else (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            f"{name} {sc:.2f}",
            (int(x1), int(y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return frame, list(zip(boxes, scores, clses))


class AlarmDispatcher:
    """Dispatch frame detections to relay and optional remote alarm output."""

    def __init__(self, enabled=True, cooldown_s=None, esp_url=None):
        self.enabled = enabled
        self.hold_s = float(DEFAULT_ALARM_HOLD_S if cooldown_s is None else cooldown_s)
        self.esp_url = esp_url.strip() if esp_url else None
        self.relay = None
        self.relay_error = None
        self._warned_unavailable = False
        self._job_queue = queue.Queue(maxsize=16)
        self._stop_event = threading.Event()
        self._worker = None
        self._active_species = set()
        self._last_seen_at = {}

        if not self.enabled:
            return
        try:
            from relay_toggle import RelayController

            self.relay = RelayController()
        except Exception as exc:
            self.relay_error = repr(exc)
            print(f"[alarm] relay disabled: {exc}", flush=True)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="offline-alarm-worker")
        self._worker.start()

    def self_test(self, species="hunter"):
        if not self.enabled:
            print("[alarm] self-test skipped: alarm disabled", flush=True)
            return False
        if self.relay is None:
            print(f"[alarm] self-test skipped: relay unavailable ({self.relay_error})", flush=True)
            return False
        print(f"[alarm] self-test trigger {species}", flush=True)
        self.relay.trigger(species=species)
        return True

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                self._job_queue.task_done()
                break
            species = job.get("species")
            active = bool(job.get("active", False))
            frame_idx = job.get("frame_idx")
            if self.relay is not None:
                try:
                    print(
                        f"[alarm-worker] set relay {'ON' if active else 'OFF'} for {species} from frame={frame_idx}",
                        flush=True,
                    )
                    self.relay.set_active(active, species=species)
                except Exception as exc:
                    print(f"[alarm] relay failed for {species}: {exc}", flush=True)
            elif not self._warned_unavailable:
                self._warned_unavailable = True
                print(f"[alarm] relay unavailable, skip local trigger: {self.relay_error}", flush=True)
            if active:
                self._post_remote_alarm(species)
            self._job_queue.task_done()

    def _post_remote_alarm(self, species):
        if not self.esp_url:
            return
        params = urllib.parse.urlencode({"source": species, "pattern": species})
        url = f"{self.esp_url}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                body = response.read().decode("utf-8", "ignore").strip()
            print(f"[alarm] remote sent for {species}: {body or 'ok'}", flush=True)
        except urllib.error.URLError as exc:
            print(f"[alarm] remote failed for {species}: {exc}", flush=True)

    def handle_detections(self, dets, frame_idx):
        if not self.enabled:
            return
        now = time.time()
        species_in_frame = set()
        for _, score, cls_idx in dets:
            cls_idx = int(cls_idx)
            if cls_idx < 0 or cls_idx >= len(CLASSES):
                continue
            species = CLASSES[cls_idx]
            if species in ALARM_SPECIES:
                species_in_frame.add(species)
                self._last_seen_at[species] = now

        for species in sorted(ALARM_SPECIES):
            if species in species_in_frame:
                if species in self._active_species:
                    continue
                self._active_species.add(species)
                print(f"[alarm] frame={frame_idx} queue relay ON for {species}", flush=True)
                try:
                    self._job_queue.put_nowait({"species": species, "frame_idx": frame_idx, "active": True})
                except queue.Full:
                    print(f"[alarm] worker queue full, drop relay update for {species} frame={frame_idx}", flush=True)
                continue

            if species not in self._active_species:
                continue
            last_seen = self._last_seen_at.get(species, 0.0)
            if now - last_seen < self.hold_s:
                continue
            self._active_species.discard(species)
            print(f"[alarm] frame={frame_idx} queue relay OFF for {species} after {self.hold_s:.1f}s idle", flush=True)
            try:
                self._job_queue.put_nowait({"species": species, "frame_idx": frame_idx, "active": False})
            except queue.Full:
                print(f"[alarm] worker queue full, drop relay update for {species} frame={frame_idx}", flush=True)

    def close(self):
        self._stop_event.set()
        if self._worker is not None:
            try:
                self._job_queue.put_nowait(None)
            except queue.Full:
                pass
            self._worker.join(timeout=1.0)
            self._worker = None
        if self.relay is not None:
            try:
                self.relay.close()
            except Exception:
                pass
            self.relay = None


def parse_args():
    parser = argparse.ArgumentParser(description="Offline video inference with relay/alarm output")
    parser.add_argument("--model", default=MODEL, help="BPU model path")
    parser.add_argument("--video", default=VIDEO, help="Input video path")
    parser.add_argument("--output", default=OUTPUT, help="Output annotated video path")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_ALARM_HOLD_S, help="Relay hold seconds after last detection")
    parser.add_argument("--esp-url", default=os.environ.get("FY_REMOTE_ALARM_ESP", ""), help="Optional remote alarm URL")
    parser.add_argument("--no-relay", action="store_true", help="Disable relay/alarm output and only run inference")
    parser.add_argument(
        "--relay-self-test",
        action="store_true",
        help="Trigger one relay self-test at startup before inference loop",
    )
    parser.add_argument(
        "--relay-self-test-species",
        default="hunter",
        choices=sorted(ALARM_SPECIES),
        help="Species mapping used for relay self-test",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] load model: {args.model}", flush=True)
    model = dnn.load(args.model)

    print(f"[INFO] open video: {args.video}", flush=True)
    # 摄像头模式: 强制 V4L2 backend (obsensor backend 在 RDK X5 上不能 index 打开)
    if args.video.lstrip('-').isdigit():
        cam_idx = int(args.video)
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(args.video)
    # 摄像头模式: 显式设 MJPG + 640x480 + 30fps (避免 cv2 默认协商失败 width=-1)
    if args.video.isdigit() or args.video.lstrip('-').isdigit():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("[ERROR] unable to open input video", flush=True)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height} @ {fps_in:.2f} fps, total={total}", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))
    if not writer.isOpened():
        print("[ERROR] unable to create output video", flush=True)
        cap.release()
        return 1

    dispatcher = AlarmDispatcher(
        enabled=not args.no_relay,
        cooldown_s=args.cooldown,
        esp_url=args.esp_url,
    )

    print(f"[INFO] output: {args.output}", flush=True)
    if args.no_relay:
        print("[INFO] relay/alarm disabled by --no-relay", flush=True)
    else:
        print(f"[INFO] relay/alarm enabled, state mode=hold-{args.cooldown:.1f}s-after-last-detection", flush=True)
        print(f"[INFO] alarm species: {sorted(ALARM_SPECIES)}", flush=True)
        if dispatcher.relay_error:
            print(f"[WARN] relay unavailable: {dispatcher.relay_error}", flush=True)
        if args.esp_url:
            print(f"[INFO] remote alarm url: {args.esp_url}", flush=True)
        if args.relay_self_test:
            dispatcher.self_test(args.relay_self_test_species)
    print("-" * 50, flush=True)

    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame, dets = infer_frame(model, frame, height, width)
            # === HDMI 实时显示 (DISPLAY=:0 弹窗) ===
            if frame_count == 0:
                cv2.namedWindow('live_infer', cv2.WINDOW_NORMAL)
                # === HDMI 全屏 (WND_PROP_FULLSCREEN=0, propValue=1.0 表示 fullscreen) ===
                cv2.setWindowProperty('live_infer', cv2.WND_PROP_FULLSCREEN, 1.0)
            cv2.imshow('live_infer', out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('[INFO] pressed q, exiting', flush=True); break
            dispatcher.handle_detections(dets, frame_count + 1)
            writer.write(out_frame)
            frame_count += 1
            if frame_count % 10 == 0 or frame_count == 1:
                elapsed = time.time() - t0
                cur_fps = frame_count / elapsed if elapsed > 0 else 0.0
                summary = "none"
                if dets:
                    names = []
                    for _, score, cls_idx in dets:
                        cls_idx = int(cls_idx)
                        if 0 <= cls_idx < len(CLASSES):
                            names.append(f"{CLASSES[cls_idx]}:{float(score):.2f}")
                    summary = ", ".join(names) if names else f"{len(dets)} obj(s)"
                print(f"  [{frame_count}/{total}] {summary} | {cur_fps:.1f} fps", flush=True)
    finally:
        cap.release()
        writer.release()
        dispatcher.close()
        try: cv2.destroyAllWindows()
        except: pass

    total_time = time.time() - t0
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    print("-" * 50, flush=True)
    print(f"[DONE] {frame_count}/{total} frames | {total_time:.1f}s | {avg_fps:.1f} fps", flush=True)
    print(f"[DONE] output: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
