#!/usr/bin/env python3
"""Bridge local alarm events to an ESP8266 buzzer over Wi-Fi.

This script does not change the existing recognition logic. It only polls the
already exposed local APIs and forwards new alarm events to an ESP8266 HTTP
receiver.

Default local endpoints:
  - http://127.0.0.1:5000/orchestrator/health
  - http://127.0.0.1:5000/alarm

Default ESP endpoint:
  - http://192.168.4.1/alarm
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_LOCAL_BASE = "http://127.0.0.1:5000"
DEFAULT_ESP_URL = "http://192.168.4.1/alarm"
DEFAULT_POLL_INTERVAL = 0.35
DEFAULT_HTTP_TIMEOUT = 1.0

SPECIES_PATTERNS = {
    "hunter": "hunter",
    "gun": "hunter",
    "snake": "snake",
    "weasel": "weasel",
    "pressure": "pressure",
}

REMOTE_SOURCE_MAP = {
    "hunter": "pressure",
    "gun": "pressure",
}

DEFAULT_DURATION_MS = {
    "hunter": 5000,
    "snake": 3500,
    "weasel": 2800,
    "pressure": 1800,
}


def fetch_json(url: str, timeout: float) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[remote_alarm] fetch failed: {url} -> {exc}", flush=True)
        return None


def post_alarm(
    esp_url: str,
    source: str,
    timeout: float,
    *,
    cells: Optional[Iterable[str]] = None,
    duration_ms: Optional[int] = None,
) -> bool:
    remote_source = REMOTE_SOURCE_MAP.get(source, source)
    effective_duration_ms = duration_ms or DEFAULT_DURATION_MS.get(remote_source, DEFAULT_DURATION_MS.get(source, 2000))
    params = {
        "source": remote_source,
        "pattern": SPECIES_PATTERNS.get(source, source),
        "duration_ms": str(effective_duration_ms),
    }
    if cells:
        params["cells"] = ",".join(cells)
    url = f"{esp_url}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "ignore").strip()
        print(f"[remote_alarm] sent {source} -> {remote_source}: {body or 'ok'}", flush=True)
        return True
    except Exception as exc:
        print(f"[remote_alarm] send failed: {url} -> {exc}", flush=True)
        return False


def grid_to_cells(grid: List[List[int]]) -> Set[str]:
    active: Set[str] = set()
    for row_idx, row in enumerate(grid):
        for col_idx, value in enumerate(row):
            if value:
                active.add(f"{chr(ord('A') + row_idx)}{col_idx + 1}")
    return active


def extract_relay_timestamps(health: dict) -> Dict[str, float]:
    fusion = (health or {}).get("fusion") or {}
    last_relay_at = fusion.get("last_relay_at") or {}
    result: Dict[str, float] = {}
    for species in ("hunter", "gun", "snake", "weasel"):
        try:
            result[species] = float(last_relay_at.get(species, 0.0) or 0.0)
        except (TypeError, ValueError):
            result[species] = 0.0
    return result


def loop(local_base: str, esp_url: str, poll_interval: float, timeout: float) -> None:
    health_url = f"{local_base.rstrip('/')}/orchestrator/health"
    pressure_url = f"{local_base.rstrip('/')}/alarm"

    last_species_ts: Dict[str, float] = {"hunter": 0.0, "gun": 0.0, "snake": 0.0, "weasel": 0.0}
    previous_pressure_cells: Set[str] = set()

    print(f"[remote_alarm] health_url={health_url}", flush=True)
    print(f"[remote_alarm] pressure_url={pressure_url}", flush=True)
    print(f"[remote_alarm] esp_url={esp_url}", flush=True)

    while True:
        health = fetch_json(health_url, timeout)
        if health is not None:
            relay_timestamps = extract_relay_timestamps(health)
            for species, ts in relay_timestamps.items():
                if ts > last_species_ts.get(species, 0.0):
                    post_alarm(esp_url, species, timeout)
                    last_species_ts[species] = ts

        pressure = fetch_json(pressure_url, timeout)
        if pressure is not None:
            grid = pressure.get("grid")
            if isinstance(grid, list):
                current_cells = grid_to_cells(grid)
                new_cells = sorted(current_cells - previous_pressure_cells)
                if new_cells:
                    post_alarm(
                        esp_url,
                        "pressure",
                        timeout,
                        cells=new_cells,
                        duration_ms=DEFAULT_DURATION_MS["pressure"],
                    )
                previous_pressure_cells = current_cells

        time.sleep(poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward local alarms to ESP8266 buzzer")
    parser.add_argument("--local-base", default=DEFAULT_LOCAL_BASE, help="Base URL of the local web service")
    parser.add_argument("--esp-url", default=DEFAULT_ESP_URL, help="ESP8266 /alarm endpoint")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="Polling interval in seconds")
    parser.add_argument("--timeout", type=float, default=DEFAULT_HTTP_TIMEOUT, help="HTTP timeout in seconds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    loop(
        local_base=args.local_base,
        esp_url=args.esp_url,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
