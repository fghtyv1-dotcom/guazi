#!/usr/bin/env python3
"""Import-safe relay controller and legacy standalone toggle entry."""

import sys
import time

import Hobot.GPIO as GPIO

PINS = [11, 13]
SPECIES_PINS = {
    "snake": [11],
    "weasel": [13],
    "hunter": [11, 13],
    "gun": [11, 13],
}
HOLD_S = 0.8
CYCLES = 3


class RelayController:
    """Expose both pulse-style and state-style relay control."""

    def __init__(self, pins=None):
        self.pins = list(pins or PINS)
        self._initialized = False
        self._setup()

    def _setup(self):
        if self._initialized:
            return
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pins, GPIO.OUT, initial=GPIO.LOW)
        self._initialized = True
        print(f"[relay_toggle] Pin {self.pins} configured as OUTPUT, initial LOW")
        sys.stdout.flush()

    def _resolve_pins(self, pins=None, species=None):
        return list(pins or SPECIES_PINS.get(species, self.pins))

    def set_active(self, active=True, pins=None, species=None):
        active_pins = self._resolve_pins(pins=pins, species=species)
        level = GPIO.HIGH if active else GPIO.LOW
        state = "HIGH" if active else "LOW"
        for pin in active_pins:
            GPIO.output(pin, level)
        print(f"[relay_toggle] Set {active_pins} -> {state} ({species or 'manual'})")
        sys.stdout.flush()

    def trigger(self, cycles=CYCLES, hold_s=HOLD_S, pins=None, species=None):
        active_pins = self._resolve_pins(pins=pins, species=species)
        print(f"[relay_toggle] Start {cycles} cycles on {active_pins} ({hold_s}s HIGH + {hold_s}s LOW)")
        sys.stdout.flush()
        for i in range(1, cycles + 1):
            self.set_active(True, pins=active_pins, species=species)
            print(f"  cycle {i}/{cycles} -> HIGH")
            sys.stdout.flush()
            time.sleep(hold_s)

            self.set_active(False, pins=active_pins, species=species)
            print(f"  cycle {i}/{cycles} -> LOW")
            sys.stdout.flush()
            time.sleep(hold_s)

    def close(self):
        if not self._initialized:
            return
        for pin in self.pins:
            try:
                GPIO.output(pin, GPIO.LOW)
            except Exception:
                pass
        GPIO.cleanup()
        self._initialized = False
        print(f"[relay_toggle] Exit, Pin {self.pins} set LOW")
        sys.stdout.flush()


def main():
    controller = RelayController()
    try:
        controller.trigger()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
