#!/usr/bin/env python3
"""Resolve drifting USB devices without changing business logic."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterable, Optional, Tuple


SERIAL_NAME_HINTS = ("ch340", "wch", "arduino", "usb-serial", "1a86", "7523")
CAMERA_NAME_HINTS = ("usb", "camera", "webcam", "uvc", "video")
AUDIO_NAME_HINTS = ("respeaker", "xvf3800", "seeed", "mic array", "usb")


def _existing_paths(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        if path and os.path.exists(path):
            yield path


def _parse_device_token(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    if "," in text:
        parts = [part.strip() for part in text.split(",", 1)]
        if len(parts) == 2 and all(part.lstrip("-").isdigit() for part in parts):
            return int(parts[0]), int(parts[1])
    return text


def resolve_serial_port(preferred: Optional[str] = None) -> Optional[str]:
    env_preferred = os.environ.get("FY_UART_PORT")
    for candidate in _existing_paths(
        [
            env_preferred,
            preferred,
            "/dev/mega0",
        ]
    ):
        return candidate

    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    if by_id:
        scored = sorted(
            by_id,
            key=lambda path: (
                0 if any(h in os.path.basename(path).lower() for h in SERIAL_NAME_HINTS) else 1,
                path,
            ),
        )
        return scored[0]

    try:
        from serial.tools import list_ports

        matches = []
        for port in list_ports.comports():
            score = 100
            if port.vid == 0x1A86 and port.pid == 0x7523:
                score = 0
            else:
                blob = " ".join(
                    str(part or "")
                    for part in (port.device, port.name, port.description, port.hwid, port.manufacturer)
                ).lower()
                if any(h in blob for h in SERIAL_NAME_HINTS):
                    score = 10
            matches.append((score, port.device))
        matches.sort()
        if matches and matches[0][1]:
            return matches[0][1]
    except Exception:
        pass

    fallback = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    return fallback[0] if fallback else preferred


def resolve_camera_device(preferred: Any = None) -> Any:
    env_preferred = _parse_device_token(os.environ.get("FY_CAMERA_DEVICE"))
    preferred = _parse_device_token(env_preferred if env_preferred is not None else preferred)

    if isinstance(preferred, str) and os.path.exists(preferred):
        return preferred

    by_id = sorted(glob.glob("/dev/v4l/by-id/*"))
    if by_id:
        scored = sorted(
            by_id,
            key=lambda path: (
                0 if "index0" in os.path.basename(path).lower() else 1,
                0 if any(h in os.path.basename(path).lower() for h in CAMERA_NAME_HINTS) else 1,
                path,
            ),
        )
        return scored[0]

    by_path = sorted(glob.glob("/dev/v4l/by-path/*"))
    if by_path:
        return by_path[0]

    return 0 if preferred is None else preferred


def describe_camera_device(device: Any) -> Dict[str, Any]:
    path = None
    if isinstance(device, str):
        path = device
    elif isinstance(device, int):
        guess = f"/dev/video{device}"
        if os.path.exists(guess):
            path = guess
    return {
        "device": device,
        "path_exists": bool(path and os.path.exists(path)),
        "real_path": os.path.realpath(path) if path and os.path.exists(path) else None,
    }


def resolve_audio_input_device(preferred: Any = None) -> Tuple[Any, Dict[str, Any]]:
    preferred = _parse_device_token(os.environ.get("FY_AUDIO_DEVICE", preferred))

    try:
        import sounddevice as sd
    except Exception as exc:
        return preferred, {"error": f"sounddevice_import_failed:{exc}"}

    if os.environ.get("ALSA_DEVICE"):
        alsa_choice = _parse_device_token(os.environ["ALSA_DEVICE"])
        return alsa_choice, {"source": "ALSA_DEVICE", "device": alsa_choice}

    info: Dict[str, Any] = {"source": "auto", "device": preferred}
    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        info["error"] = f"query_failed:{exc}"
        return preferred, info

    candidates = []
    for index, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0) or 0)
        if max_in <= 0:
            continue
        name = str(dev.get("name", ""))
        lower_name = name.lower()
        score = 100
        if max_in >= 2:
            score -= 5
        if isinstance(preferred, str) and preferred.lower() in lower_name:
            score = 0
        elif any(h in lower_name for h in AUDIO_NAME_HINTS):
            score = 10
        elif "input" in lower_name or "mic" in lower_name:
            score = 30
        candidates.append(
            (
                score,
                index,
                {
                    "index": index,
                    "name": name,
                    "max_input_channels": max_in,
                    "default_samplerate": dev.get("default_samplerate"),
                },
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1]))

    if candidates:
        best_score, best_index, best_meta = candidates[0]
        if best_score < 100 or preferred is None:
            info.update(best_meta)
            info["score"] = best_score
            info["candidates_checked"] = len(candidates)
            return best_index, info

    if isinstance(preferred, (int, tuple)):
        info["source"] = "preferred"
        return preferred, info

    default_input = None
    try:
        default_device = sd.default.device
        if isinstance(default_device, (list, tuple)) and default_device:
            default_input = default_device[0]
    except Exception:
        default_input = None

    if default_input not in (None, -1):
        info["source"] = "default"
        info["device"] = default_input
        return default_input, info

    return preferred, info


def describe_audio_input_device(device: Any) -> Dict[str, Any]:
    resolved, info = resolve_audio_input_device(preferred=device)
    info = dict(info)
    info["device"] = resolved
    return info
