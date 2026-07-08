#!/usr/bin/env python3
"""Unified event-driven entrypoint for the audio, vision, and relay modules."""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import cv2
import serial

from audio_adapter import AudioWorker
from device_resolver import describe_camera_device, resolve_camera_device, resolve_serial_port
from relay_toggle import CYCLES as RELAY_CYCLES
from relay_toggle import HOLD_S as RELAY_HOLD_S
from relay_toggle import RelayController
import visual_track

TRACK_MODE_PATH = "/tmp/tracking_mode.json"
VISION_SERVO_PRIORITY = 30
RELAY_PRIORITY = 5
HUNTER_RELAY_PRIORITY = 1
RELAY_COOLDOWN_S = 5.0
RELAY_VISION_TTL_S = 5.0
RELAY_AUDIO_TTL_S = 5.0
RELAY_REFRESH_INTERVAL_S = 0.5
HEARTBEAT_INTERVAL_S = 10.0
ALARM_SERVER_PORT = 8001
REMOTE_ALARM_ESP_URL = "http://192.168.4.1/alarm"
PUBLIC_DASHBOARD_HOST = os.environ.get("FY_DASHBOARD_HOST", "192.168.4.100")
PUBLIC_DASHBOARD_URL = f"http://{PUBLIC_DASHBOARD_HOST}:{visual_track.STREAM_PORT}/"


def should_enable_local_display():
    mode = os.environ.get("FY_LOCAL_DISPLAY", "auto").strip().lower()
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode in {"1", "true", "on", "yes"}:
        return True
    if sys.platform.startswith("win"):
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


LOCAL_DISPLAY_ENABLED = should_enable_local_display()
LOCAL_DASHBOARD_URL = PUBLIC_DASHBOARD_URL


def put_latest(target_queue, item):
    try:
        target_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    target_queue.put_nowait(item)


def utc_ts():
    return time.time()


def safe_qsize(target_queue):
    try:
        return target_queue.qsize()
    except Exception:
        return -1


def age_seconds(timestamp):
    if not timestamp:
        return None
    return round(max(0.0, utc_ts() - timestamp), 2)


def read_tracking_mode(default="video"):
    try:
        with open(TRACK_MODE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh).get("TRACK_MODE", default)
    except Exception:
        return default


def ensure_tracking_mode(default="video"):
    try:
        with open(TRACK_MODE_PATH, "r", encoding="utf-8") as fh:
            json.load(fh)
            return
    except Exception:
        pass

    try:
        with open(TRACK_MODE_PATH, "w", encoding="utf-8") as fh:
            json.dump({"TRACK_MODE": default}, fh)
    except Exception:
        pass


class ManagedWorkerThread(threading.Thread):
    def __init__(self, name, target):
        super().__init__(daemon=True, name=name)
        self._target_func = target
        self.started_at = None
        self.stopped_at = None
        self.last_error = None

    def run(self):
        self.started_at = utc_ts()
        print(f"[thread] {self.name} starting", flush=True)
        try:
            self._target_func()
        except Exception as exc:
            self.last_error = repr(exc)
            print(f"[thread] {self.name} failed: {exc}", flush=True)
            traceback.print_exc()
            raise
        finally:
            self.stopped_at = utc_ts()
            print(f"[thread] {self.name} stopped", flush=True)

    def get_status(self):
        return {
            "name": self.name,
            "alive": self.is_alive(),
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_error": self.last_error,
        }


class SharedCamera(threading.Thread):
    def __init__(self, stop_event, camera_id=None):
        super().__init__(daemon=True, name="camera-producer")
        self.stop_event = stop_event
        self.camera_preference = visual_track.CAMERA_ID if camera_id is None else camera_id
        self.camera_id = self.camera_preference
        self.capture = None
        self.started_at = None
        self.last_frame_at = None
        self.frame_count = 0
        self.read_failures = 0
        self.last_error = None
        self.last_frame_shape = None

    def get_status(self):
        return {
            "alive": self.is_alive(),
            "camera_id": self.camera_id,
            "camera_preference": self.camera_preference,
            "started_at": self.started_at,
            "last_frame_at": self.last_frame_at,
            "frame_age_s": age_seconds(self.last_frame_at),
            "frame_count": self.frame_count,
            "read_failures": self.read_failures,
            "last_frame_shape": self.last_frame_shape,
            "last_error": self.last_error,
            "resolved": describe_camera_device(self.camera_id),
        }

    def run(self):
        self.started_at = utc_ts()
        try:
            self.camera_id = resolve_camera_device(self.camera_preference)
            visual_track.CAMERA_ID = self.camera_id
            self.capture = cv2.VideoCapture(self.camera_id)
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            if not self.capture.isOpened():
                raise RuntimeError(f"camera {self.camera_id} could not be opened")

            print(f"[camera] producer started on camera {self.camera_id}", flush=True)
            for _ in range(5):
                self.capture.read()

            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    self.read_failures += 1
                    time.sleep(0.05)
                    continue
                self.last_frame_at = utc_ts()
                self.frame_count += 1
                self.last_frame_shape = list(frame.shape)
                visual_track.push_camera_frame(frame)
        except Exception as exc:
            self.last_error = repr(exc)
            print(f"[camera] producer failed: {exc}", flush=True)
            raise
        finally:
            if self.capture is not None:
                self.capture.release()


class RelayWorker(threading.Thread):
    def __init__(self, stop_event):
        super().__init__(daemon=True, name="relay-worker")
        self.stop_event = stop_event
        self.command_queue = queue.Queue(maxsize=16)
        self.controller = None
        self.started_at = None
        self.last_command_at = None
        self.last_trigger_at = None
        self.last_command = None
        self.commands_processed = 0
        self.last_error = None
        self.source_states = {}
        self.source_expiry = {}
        self.effective_states = {}

    def submit(self, command):
        put_latest(self.command_queue, command)

    def get_status(self):
        return {
            "alive": self.is_alive(),
            "started_at": self.started_at,
            "queue_size": safe_qsize(self.command_queue),
            "last_command_at": self.last_command_at,
            "last_trigger_at": self.last_trigger_at,
            "last_command": self.last_command,
            "commands_processed": self.commands_processed,
            "last_error": self.last_error,
            "effective_states": dict(self.effective_states),
            "source_states": {species: dict(sources) for species, sources in self.source_states.items()},
        }

    def _set_source_state(self, species, source, active, ttl_s=None):
        species_sources = self.source_states.setdefault(species, {})
        species_expiry = self.source_expiry.setdefault(species, {})
        if active:
            species_sources[source] = True
            if ttl_s is not None:
                species_expiry[source] = utc_ts() + float(ttl_s)
            else:
                species_expiry.pop(source, None)
        else:
            species_sources.pop(source, None)
            species_expiry.pop(source, None)
            if not species_sources:
                self.source_states.pop(species, None)
            if not species_expiry:
                self.source_expiry.pop(species, None)

        new_effective = bool(self.source_states.get(species))
        old_effective = bool(self.effective_states.get(species))
        if new_effective == old_effective:
            return
        self.effective_states[species] = new_effective
        self.controller.set_active(new_effective, species=species)
        self.last_trigger_at = utc_ts()
        print(f"[relay] state {'ON' if new_effective else 'OFF'} for {species} (source={source})", flush=True)

    def _expire_ttl_states(self):
        now = utc_ts()
        expired = []
        for species, sources in list(self.source_expiry.items()):
            for source, expires_at in list(sources.items()):
                if expires_at and now >= expires_at:
                    expired.append((species, source))
        for species, source in expired:
            self._set_source_state(species, source, False, ttl_s=None)

    def run(self):
        self.started_at = utc_ts()
        self.controller = RelayController()
        try:
            while not self.stop_event.is_set():
                try:
                    command = self.command_queue.get(timeout=0.2)
                except queue.Empty:
                    self._expire_ttl_states()
                    continue
                self._expire_ttl_states()
                if command.get("kind") not in {"relay", "relay_state"}:
                    continue
                self.last_command_at = utc_ts()
                self.last_command = dict(command)
                if command.get("kind") == "relay_state":
                    self._set_source_state(
                        command.get("species"),
                        command.get("source", "unknown"),
                        bool(command.get("active", False)),
                        ttl_s=command.get("ttl_s"),
                    )
                else:
                    self.controller.trigger(
                        cycles=int(command.get("cycles", RELAY_CYCLES)),
                        hold_s=float(command.get("hold_s", RELAY_HOLD_S)),
                        species=command.get("species"),
                        pins=command.get("pins"),
                    )
                    self.last_trigger_at = utc_ts()
                self.commands_processed += 1
        except Exception as exc:
            self.last_error = repr(exc)
            print(f"[relay] worker failed: {exc}", flush=True)
            raise
        finally:
            if self.controller is not None:
                self.controller.close()


class ServoController:
    def __init__(self, port=None, baud=None):
        self.port = port or visual_track.PARAMS["UART_PORT"]
        self.port_preference = self.port
        self.baud = baud or visual_track.PARAMS["UART_BAUD"]
        self.serial_conn = None
        self.lock = threading.Lock()
        self.reader_started = False
        self.connect_attempts = 0
        self.connected_at = None
        self.last_move_at = None
        self.last_angle = None
        self.move_count = 0
        self.last_error = None
        self.last_connect_attempt_at = None
        self.reconnect_interval_s = 1.0
        self.startup_grace_s = 8.0
        self.stale_rx_timeout_s = 15.0

    def get_status(self):
        return {
            "connected": self.serial_conn is not None,
            "port": self.port,
            "port_preference": self.port_preference,
            "baud": self.baud,
            "connect_attempts": self.connect_attempts,
            "connected_at": self.connected_at,
            "last_move_at": self.last_move_at,
            "last_angle": self.last_angle,
            "move_count": self.move_count,
            "last_error": self.last_error,
            "reader_started": self.reader_started,
            "last_connect_attempt_at": self.last_connect_attempt_at,
        }

    def _mark_disconnected(self, reason=None):
        if reason:
            self.last_error = reason
            print(f"[control] servo disconnected: {reason}", flush=True)
        try:
            if self.serial_conn is not None:
                self.serial_conn.close()
        except Exception:
            pass
        self.serial_conn = None
        visual_track.motor = None
        visual_track.reset_mega_runtime_state()

    def connect(self, force=False):
        now = utc_ts()
        if self.serial_conn is not None and not force:
            return True
        if (not force and self.last_connect_attempt_at is not None
                and now - self.last_connect_attempt_at < self.reconnect_interval_s):
            return False
        self.last_connect_attempt_at = now
        self.connect_attempts += 1
        if force:
            self._mark_disconnected()
        try:
            resolved_port = resolve_serial_port(self.port_preference or self.port)
            self.port = resolved_port or self.port_preference or self.port
            if not self.port:
                self._mark_disconnected("serial device not found")
                print("[control] servo unavailable: serial device not found", flush=True)
                return False
            visual_track.PARAMS["UART_PORT"] = self.port
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0.3)
            self.connected_at = utc_ts()
            visual_track.motor = self.serial_conn
            visual_track.reset_mega_runtime_state()
            self.last_error = None
            print(f"[control] servo connected: {self.port} @ {self.baud}", flush=True)
            if not self.reader_started:
                self.reader_started = True
                threading.Thread(target=visual_track._serial_reader_loop, daemon=True, name="mega-reader").start()
            return True
        except Exception as exc:
            self._mark_disconnected(repr(exc))
            print(f"[control] servo unavailable: {exc}", flush=True)
            return False

    def ensure_connected(self):
        if self.serial_conn is None:
            return self.connect()
        if not getattr(self.serial_conn, "is_open", True):
            self._mark_disconnected("serial port closed")
            return self.connect()
        now = utc_ts()
        connected_at = self.connected_at or now
        last_rx = getattr(visual_track, "mega_last_rx_at", None)
        if last_rx is None:
            if now - connected_at > self.startup_grace_s:
                self._mark_disconnected(f"no_rx_after_connect>{self.startup_grace_s}s")
                return self.connect()
            return True
        if now - last_rx > self.stale_rx_timeout_s:
            self._mark_disconnected(f"stale_rx>{self.stale_rx_timeout_s}s")
            return self.connect()
        return True

    def move(self, angle):
        if not self.ensure_connected():
            return
        if getattr(visual_track, "mega_boot_at", None) is not None and getattr(visual_track, "mega_homed_at", None) is None:
            print(f"[control] servo move skipped during zero lock: angle={angle}", flush=True)
            return
        with self.lock:
            try:
                self.serial_conn.write(f"T{int(round(angle))}\n".encode())
                visual_track.current_angle = float(angle)
                self.last_move_at = utc_ts()
                self.last_angle = float(angle)
                self.move_count += 1
            except Exception as exc:
                self._mark_disconnected(repr(exc))
                print(f"[control] servo write failed: {exc}", flush=True)

    def close(self):
        self._mark_disconnected()


class ControlLayer(threading.Thread):
    def __init__(self, stop_event, control_queue, relay_worker):
        super().__init__(daemon=True, name="control-layer")
        self.stop_event = stop_event
        self.control_queue = control_queue
        self.relay_worker = relay_worker
        self.servo = ServoController()
        self.started_at = None
        self.last_command_at = None
        self.last_command = None
        self.commands_processed = 0
        self.last_error = None

    def get_status(self):
        return {
            "alive": self.is_alive(),
            "started_at": self.started_at,
            "queue_size": safe_qsize(self.control_queue),
            "last_command_at": self.last_command_at,
            "last_command": self.last_command,
            "commands_processed": self.commands_processed,
            "last_error": self.last_error,
            "servo": self.servo.get_status(),
        }

    def run(self):
        self.started_at = utc_ts()
        self.servo.connect()
        try:
            while not self.stop_event.is_set():
                try:
                    _, _, command = self.control_queue.get(timeout=0.2)
                except queue.Empty:
                    self.servo.ensure_connected()
                    continue

                self.last_command_at = utc_ts()
                self.last_command = dict(command)
                kind = command.get("kind")
                if kind == "servo":
                    self.servo.move(command["angle"])
                elif kind in {"relay", "relay_state"}:
                    self.relay_worker.submit(command)
                self.commands_processed += 1
        except Exception as exc:
            self.last_error = repr(exc)
            print(f"[control] layer failed: {exc}", flush=True)
            raise
        finally:
            self.servo.close()


class FusionEngine(threading.Thread):
    def __init__(self, stop_event, event_queue, control_queue):
        super().__init__(daemon=True, name="fusion-layer")
        self.stop_event = stop_event
        self.event_queue = event_queue
        self.control_queue = control_queue
        self.last_relay_at = {}
        self.started_at = None
        self.last_event_at = None
        self.last_event = None
        self.event_counts = {}
        self.last_error = None
        self.relay_source_state = {}
        self.relay_effective_state = {}
        self.relay_source_refresh_at = {}

    def get_status(self):
        return {
            "alive": self.is_alive(),
            "started_at": self.started_at,
            "last_event_at": self.last_event_at,
            "last_event": self.last_event,
            "event_counts": dict(self.event_counts),
            "last_relay_at": dict(self.last_relay_at),
            "relay_effective_state": dict(self.relay_effective_state),
            "last_error": self.last_error,
        }

    def enqueue_control(self, priority, command):
        put_latest(self.control_queue, (priority, utc_ts(), command))

    def _set_relay_state(self, species, source, active, ttl_s=None):
        now = utc_ts()
        species_sources = self.relay_source_state.setdefault(species, {})
        prev_source_active = source in species_sources
        prev_effective = bool(self.relay_effective_state.get(species))
        if active:
            species_sources[source] = True
        else:
            species_sources.pop(source, None)
            if not species_sources:
                self.relay_source_state.pop(species, None)
        new_effective = bool(self.relay_source_state.get(species))
        self.relay_effective_state[species] = new_effective
        refresh_key = (species, source)
        if active and ttl_s is not None and prev_source_active:
            last_refresh = self.relay_source_refresh_at.get(refresh_key, 0.0)
            if now - last_refresh < RELAY_REFRESH_INTERVAL_S:
                return
        if ttl_s is None and prev_source_active == active:
            return
        relay_priority = HUNTER_RELAY_PRIORITY if species in {"hunter", "gun"} else RELAY_PRIORITY
        command = {
            "kind": "relay_state",
            "species": species,
            "source": source,
            "active": bool(active),
        }
        if active and ttl_s is not None:
            command["ttl_s"] = float(ttl_s)
        self.enqueue_control(
            relay_priority,
            command,
        )
        if active:
            self.relay_source_refresh_at[refresh_key] = now
        else:
            self.relay_source_refresh_at.pop(refresh_key, None)
        if active and not prev_effective and new_effective:
            self.last_relay_at[species] = now
            print(f"[fusion] relay ON queued for {species} from {source}", flush=True)
        elif not active and prev_effective and not new_effective:
            print(f"[fusion] relay OFF queued for {species} from {source}", flush=True)

    def _handle_audio(self, event):
        species = event.get("species")
        if species in {"snake", "weasel", "gun"}:
            self._set_relay_state(species, "audio", True, ttl_s=RELAY_AUDIO_TTL_S)

        if read_tracking_mode() == "audio":
            self.enqueue_control(
                int(event.get("priority", 10)),
                {
                    "kind": "servo",
                    "source": "audio",
                    "angle": float(event["target_angle"]),
                },
            )

    def _handle_vision(self, event):
        classes = {det["cls_name"] for det in event.get("detections", [])}
        for species in ("hunter", "snake", "weasel"):
            if species in classes:
                self._set_relay_state(species, "vision", True, ttl_s=RELAY_VISION_TTL_S)

    def _handle_servo_intent(self, event):
        self.enqueue_control(
            int(event.get("priority", VISION_SERVO_PRIORITY)),
            {
                "kind": "servo",
                "source": event.get("source", "vision"),
                "angle": float(event["angle"]),
            },
        )

    def run(self):
        self.started_at = utc_ts()
        try:
            while not self.stop_event.is_set():
                try:
                    event = self.event_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                event_type = event.get("type", "unknown")
                self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
                self.last_event_at = utc_ts()
                self.last_event = {
                    "type": event_type,
                    "source": event.get("source"),
                    "species": event.get("species"),
                    "timestamp": event.get("timestamp"),
                }

                if event_type == "audio_detection":
                    self._handle_audio(event)
                elif event_type == "vision_detection":
                    self._handle_vision(event)
                elif event_type == "servo_intent":
                    self._handle_servo_intent(event)
        except Exception as exc:
            self.last_error = repr(exc)
            print(f"[fusion] engine failed: {exc}", flush=True)
            raise


class AlarmServerProcess:
    def __init__(self):
        self.process = None
        self.started_at = None
        self.last_error = None
        self.script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarm_server.py")

    def start(self):
        if not os.path.exists(self.script_path):
            self.last_error = f"missing:{self.script_path}"
            print(f"[alarm] server script missing: {self.script_path}", flush=True)
            return
        try:
            self.process = subprocess.Popen([sys.executable, self.script_path])
            self.started_at = utc_ts()
            print(f"[alarm] server started: {self.script_path} pid={self.process.pid}", flush=True)
        except Exception as exc:
            self.last_error = repr(exc)
            self.process = None
            print(f"[alarm] server failed to start: {exc}", flush=True)

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def get_status(self):
        return {
            "running": self.is_running(),
            "started_at": self.started_at,
            "script_path": self.script_path,
            "pid": None if self.process is None else self.process.pid,
            "last_error": self.last_error,
            "port": ALARM_SERVER_PORT,
        }

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None


class RemoteAlarmBridgeProcess:
    def __init__(self):
        self.process = None
        self.started_at = None
        self.last_error = None
        self.script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_alarm_bridge.py")
        self.local_base = f"http://127.0.0.1:{visual_track.STREAM_PORT}"
        self.esp_url = REMOTE_ALARM_ESP_URL

    def start(self):
        if not os.path.exists(self.script_path):
            self.last_error = f"missing:{self.script_path}"
            print(f"[remote-alarm] bridge script missing: {self.script_path}", flush=True)
            return
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    self.script_path,
                    "--local-base",
                    self.local_base,
                    "--esp-url",
                    self.esp_url,
                ]
            )
            self.started_at = utc_ts()
            print(
                f"[remote-alarm] bridge started: {self.script_path} pid={self.process.pid} esp={self.esp_url}",
                flush=True,
            )
        except Exception as exc:
            self.last_error = repr(exc)
            self.process = None
            print(f"[remote-alarm] bridge failed to start: {exc}", flush=True)

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def get_status(self):
        return {
            "running": self.is_running(),
            "started_at": self.started_at,
            "script_path": self.script_path,
            "pid": None if self.process is None else self.process.pid,
            "last_error": self.last_error,
            "local_base": self.local_base,
            "esp_url": self.esp_url,
        }

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None


class LocalDashboardProcess:
    def __init__(self):
        self.process = None
        self.started_at = None
        self.last_error = None
        self.browser_cmd = None
        self.url = LOCAL_DASHBOARD_URL

    def _resolve_browser(self):
        candidates = (
            "chromium-browser",
            "chromium",
            "google-chrome",
            "google-chrome-stable",
            "microsoft-edge",
            "x-www-browser",
            "xdg-open",
        )
        for candidate in candidates:
            path = shutil.which(candidate)
            if path:
                return path
        return None

    def start(self):
        if not LOCAL_DISPLAY_ENABLED:
            self.last_error = "disabled"
            print("[display] local dashboard disabled by FY_LOCAL_DISPLAY=0", flush=True)
            return
        browser = self._resolve_browser()
        if browser is None:
            self.last_error = "browser_not_found"
            print("[display] no local browser found for dashboard display", flush=True)
            return
        self.browser_cmd = browser
        try:
            args = [browser]
            browser_name = os.path.basename(browser).lower()
            if "chrom" in browser_name or "edge" in browser_name:
                args.extend(
                    [
                        "--no-first-run",
                        "--disable-session-crashed-bubble",
                        "--disable-infobars",
                        "--start-fullscreen",
                        f"--app={self.url}",
                    ]
                )
            else:
                args.append(self.url)
            self.process = subprocess.Popen(args)
            self.started_at = utc_ts()
            self.last_error = None
            print(f"[display] local dashboard started: {browser} -> {self.url}", flush=True)
        except Exception as exc:
            self.last_error = repr(exc)
            self.process = None
            print(f"[display] local dashboard failed to start: {exc}", flush=True)

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def get_status(self):
        return {
            "running": self.is_running(),
            "started_at": self.started_at,
            "pid": None if self.process is None else self.process.pid,
            "browser_cmd": self.browser_cmd,
            "url": self.url,
            "last_error": self.last_error,
            "enabled": LOCAL_DISPLAY_ENABLED,
        }

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None


class OrchestratorState:
    def __init__(self, stop_event, event_queue, control_queue):
        self.started_at = utc_ts()
        self.stop_event = stop_event
        self.event_queue = event_queue
        self.control_queue = control_queue
        self.relay_worker = None
        self.control_layer = None
        self.fusion = None
        self.camera = None
        self.audio_worker = None
        self.vision_thread = None
        self.web_thread = None
        self.heartbeat_thread = None
        self.alarm_server = None
        self.remote_alarm_bridge = None
        self.local_dashboard = None

    def _visual_status(self):
        return {
            "frame_seq": getattr(visual_track, "frame_seq", None),
            "current_fid": getattr(visual_track, "current_fid", None),
            "fps_display": getattr(visual_track, "fps_display", None),
            "last_inference_ms": getattr(visual_track, "_last_inf_ms", None),
            "last_object_count": getattr(visual_track, "_last_obj_count", None),
            "last_target_name": getattr(visual_track, "_last_target_name", None),
            "current_angle": getattr(visual_track, "current_angle", None),
            "mega_boot_at": getattr(visual_track, "mega_boot_at", None),
            "mega_homed_at": getattr(visual_track, "mega_homed_at", None),
        }

    def _thread_statuses(self):
        statuses = {}
        for name, thread_obj in (
            ("camera", self.camera),
            ("relay", self.relay_worker),
            ("control", self.control_layer),
            ("fusion", self.fusion),
            ("audio", self.audio_worker),
            ("vision", self.vision_thread),
            ("web", self.web_thread),
            ("heartbeat", self.heartbeat_thread),
        ):
            if thread_obj is None:
                statuses[name] = {"alive": False, "missing": True}
                continue
            if hasattr(thread_obj, "get_status"):
                statuses[name] = thread_obj.get_status()
            else:
                statuses[name] = {
                    "alive": thread_obj.is_alive(),
                    "name": thread_obj.name,
                }
        return statuses

    def snapshot(self):
        return {
            "system": {
                "started_at": self.started_at,
                "uptime_s": round(utc_ts() - self.started_at, 2),
                "tracking_mode": read_tracking_mode(),
                "stop_requested": self.stop_event.is_set(),
                "alarm_server_running": self.alarm_server is not None and self.alarm_server.is_running(),
                "remote_alarm_running": self.remote_alarm_bridge is not None and self.remote_alarm_bridge.is_running(),
                "local_dashboard_running": self.local_dashboard is not None and self.local_dashboard.is_running(),
            },
            "queues": {
                "event_queue": safe_qsize(self.event_queue),
                "control_queue": safe_qsize(self.control_queue),
                "relay_queue": safe_qsize(self.relay_worker.command_queue) if self.relay_worker else None,
            },
            "threads": self._thread_statuses(),
            "camera": None if self.camera is None else self.camera.get_status(),
            "relay": None if self.relay_worker is None else self.relay_worker.get_status(),
            "control": None if self.control_layer is None else self.control_layer.get_status(),
            "fusion": None if self.fusion is None else self.fusion.get_status(),
            "audio": None if self.audio_worker is None else self.audio_worker.get_status(),
            "alarm_server": None if self.alarm_server is None else self.alarm_server.get_status(),
            "remote_alarm": None if self.remote_alarm_bridge is None else self.remote_alarm_bridge.get_status(),
            "local_dashboard": None if self.local_dashboard is None else self.local_dashboard.get_status(),
            "vision": self._visual_status(),
            "runtime_threads": [thread.name for thread in threading.enumerate()],
        }


class HealthReporter(threading.Thread):
    def __init__(self, stop_event, state, interval_s=HEARTBEAT_INTERVAL_S):
        super().__init__(daemon=True, name="health-reporter")
        self.stop_event = stop_event
        self.state = state
        self.interval_s = interval_s
        self.last_report_at = None

    def get_status(self):
        return {
            "alive": self.is_alive(),
            "last_report_at": self.last_report_at,
            "interval_s": self.interval_s,
        }

    def _format_thread_states(self, snapshot):
        parts = []
        for name in ("camera", "vision", "audio", "fusion", "control", "relay", "web"):
            info = snapshot["threads"].get(name, {})
            parts.append(f"{name}={'up' if info.get('alive') else 'down'}")
        return " ".join(parts)

    def _format_queue_states(self, snapshot):
        queues = snapshot["queues"]
        return (
            f"event_q={queues.get('event_queue')} "
            f"control_q={queues.get('control_queue')} "
            f"relay_q={queues.get('relay_queue')}"
        )

    def _format_signal_states(self, snapshot):
        camera = snapshot.get("camera") or {}
        audio = snapshot.get("audio") or {}
        control = snapshot.get("control") or {}
        fusion = snapshot.get("fusion") or {}
        return (
            f"camera_frames={camera.get('frame_count')} "
            f"camera_age={camera.get('frame_age_s')}s "
            f"audio_events={audio.get('events_emitted')} "
            f"audio_last={age_seconds(audio.get('last_publish_at'))}s "
            f"fusion_last={age_seconds(fusion.get('last_event_at'))}s "
            f"servo_connected={control.get('servo', {}).get('connected')}"
        )

    def run(self):
        while not self.stop_event.is_set():
            snapshot = self.state.snapshot()
            self.last_report_at = utc_ts()
            print(
                "[health] "
                f"mode={snapshot['system']['tracking_mode']} "
                f"{self._format_thread_states(snapshot)} "
                f"{self._format_queue_states(snapshot)} "
                f"{self._format_signal_states(snapshot)}",
                flush=True,
            )
            time.sleep(self.interval_s)


def make_event_publisher(event_queue):
    def publish(event):
        put_latest(event_queue, event)

    return publish


def make_visual_motor_handler(event_queue):
    def handle(angle):
        put_latest(
            event_queue,
            {
                "type": "servo_intent",
                "source": "vision",
                "angle": float(angle),
                "priority": VISION_SERVO_PRIORITY,
                "timestamp": utc_ts(),
            },
        )

    return handle


def make_visual_event_handler(event_queue):
    def handle(event):
        payload = {"source": "vision", "timestamp": utc_ts()}
        payload.update(event)
        put_latest(event_queue, payload)

    return handle


def run_web_server():
    visual_track.app.run(
        host="0.0.0.0",
        port=visual_track.STREAM_PORT,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


def register_orchestrator_routes(state):
    if getattr(visual_track.app, "_orchestrator_routes_registered", False):
        return

    @visual_track.app.route("/orchestrator/health")
    def orchestrator_health():
        return state.snapshot()

    @visual_track.app.route("/orchestrator/threads")
    def orchestrator_threads():
        return {
            "threads": state.snapshot()["threads"],
            "runtime_threads": [thread.name for thread in threading.enumerate()],
        }

    visual_track.app._orchestrator_routes_registered = True


def log_startup_banner():
    print("[system] ===== Feiyu Orchestrator Startup =====", flush=True)
    print(f"[system] stream_port={visual_track.STREAM_PORT}", flush=True)
    print(f"[system] alarm_port={ALARM_SERVER_PORT}", flush=True)
    print(f"[system] remote_alarm_esp={REMOTE_ALARM_ESP_URL}", flush=True)
    print(f"[system] local_display={LOCAL_DISPLAY_ENABLED}", flush=True)
    print(f"[system] public_dashboard_url={PUBLIC_DASHBOARD_URL}", flush=True)
    print(f"[system] local_dashboard_url={LOCAL_DASHBOARD_URL}", flush=True)
    print(f"[system] camera_id={visual_track.CAMERA_ID}", flush=True)
    print(f"[system] uart_port={visual_track.PARAMS['UART_PORT']}", flush=True)
    print(f"[system] uart_baud={visual_track.PARAMS['UART_BAUD']}", flush=True)
    print(f"[system] tracking_mode={read_tracking_mode()}", flush=True)


def log_snapshot(prefix, snapshot):
    camera = snapshot.get("camera") or {}
    audio = snapshot.get("audio") or {}
    control = snapshot.get("control") or {}
    fusion = snapshot.get("fusion") or {}
    print(
        f"{prefix} uptime={snapshot['system']['uptime_s']}s "
        f"mode={snapshot['system']['tracking_mode']} "
        f"camera_frames={camera.get('frame_count')} "
        f"camera_age={camera.get('frame_age_s')}s "
        f"audio_events={audio.get('events_emitted')} "
        f"audio_loop_age={age_seconds(audio.get('last_loop_at'))}s "
        f"fusion_events={fusion.get('event_counts')} "
        f"servo_connected={control.get('servo', {}).get('connected')}",
        flush=True,
    )


def main():
    ensure_tracking_mode()
    log_startup_banner()

    stop_event = threading.Event()
    event_queue = queue.Queue(maxsize=256)
    control_queue = queue.PriorityQueue(maxsize=128)
    state = OrchestratorState(stop_event, event_queue, control_queue)
    alarm_server = AlarmServerProcess()
    remote_alarm_bridge = RemoteAlarmBridgeProcess()
    local_dashboard = LocalDashboardProcess()

    relay_worker = RelayWorker(stop_event)
    control_layer = ControlLayer(stop_event, control_queue, relay_worker)
    fusion = FusionEngine(stop_event, event_queue, control_queue)
    camera = SharedCamera(stop_event)
    audio_worker = AudioWorker(make_event_publisher(event_queue), stop_event=stop_event)
    vision_thread = ManagedWorkerThread("vision-worker", visual_track.main_loop)
    web_thread = ManagedWorkerThread("web-server", run_web_server)
    heartbeat_thread = HealthReporter(stop_event, state)

    state.relay_worker = relay_worker
    state.control_layer = control_layer
    state.fusion = fusion
    state.camera = camera
    state.audio_worker = audio_worker
    state.vision_thread = vision_thread
    state.web_thread = web_thread
    state.heartbeat_thread = heartbeat_thread
    state.alarm_server = alarm_server
    state.remote_alarm_bridge = remote_alarm_bridge
    state.local_dashboard = local_dashboard

    register_orchestrator_routes(state)
    visual_track.configure_orchestrator_hooks(
        motor_handler=make_visual_motor_handler(event_queue),
        event_handler=make_visual_event_handler(event_queue),
        audio_bins_provider=audio_worker.get_spectrum_snapshot,
    )

    print("[system] starting relay/control/fusion/camera/audio/vision/web threads", flush=True)
    alarm_server.start()
    relay_worker.start()
    control_layer.start()
    fusion.start()
    camera.start()
    vision_thread.start()
    audio_worker.start()
    web_thread.start()
    heartbeat_thread.start()
    remote_alarm_bridge.start()
    time.sleep(1.5)
    local_dashboard.start()

    print("[system] orchestrator started", flush=True)
    print(f"[system] video stream: http://{PUBLIC_DASHBOARD_HOST}:{visual_track.STREAM_PORT}/video_feed", flush=True)
    print(f"[system] dashboard: {PUBLIC_DASHBOARD_URL}", flush=True)
    print(f"[system] local dashboard: {LOCAL_DASHBOARD_URL}", flush=True)
    print(f"[system] health: http://{PUBLIC_DASHBOARD_HOST}:{visual_track.STREAM_PORT}/orchestrator/health", flush=True)
    print(f"[system] threads: http://{PUBLIC_DASHBOARD_HOST}:{visual_track.STREAM_PORT}/orchestrator/threads", flush=True)

    time.sleep(3.0)
    log_snapshot("[startup-health]", state.snapshot())

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[system] shutting down", flush=True)
    finally:
        stop_event.set()
        local_dashboard.stop()
        remote_alarm_bridge.stop()
        alarm_server.stop()


if __name__ == "__main__":
    main()
