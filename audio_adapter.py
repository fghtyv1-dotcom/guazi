#!/usr/bin/env python3
"""Adapter that runs the existing audio pipeline as an orchestrated worker."""

import threading
import time

import numpy as np
import sounddevice as sd

import integrated_main as audio_core
from device_resolver import resolve_audio_input_device


class AudioWorker(threading.Thread):
    """Keep the original audio recognition flow, but emit events instead of driving hardware."""

    SPECTRUM_BINS = 32
    SPECTRUM_MIN_DB = -80.0
    SPECTRUM_MAX_DB = -18.0
    SPECTRUM_SMOOTHING = 0.65
    DIAG_LOG_INTERVAL_S = 2.0

    def __init__(self, event_callback, stop_event=None, input_device=(1, 0)):
        super().__init__(daemon=True, name="audio-worker")
        self.event_callback = event_callback
        self.stop_event = stop_event or threading.Event()
        self.input_device = input_device
        self.input_device_preference = input_device
        self.input_device_info = {}
        self._audio_lock = threading.Lock()
        self._audio_buf = np.zeros(int(2.0 * audio_core.RATE), dtype=np.float32)
        self._prev_quiet = True
        self._skip_analyze = 0
        self._audio_stream = None
        self._latest_chunk = np.zeros(int(audio_core.RATE * 0.1), dtype=np.float32)
        self._latest_chunk_seq = 0
        self._spectrum_bins = [0.0] * self.SPECTRUM_BINS
        self._spectrum_smoothed = np.zeros(self.SPECTRUM_BINS, dtype=np.float32)
        self._spectrum_cache_seq = -1
        self._spectrum_updated_at = None
        self._doa_smooth = None
        self._doa_angle_prev = None
        self._last_angle = -1
        self._last_species_time = {}
        self.started_at = None
        self.stream_started_at = None
        self.last_loop_at = None
        self.last_publish_at = None
        self.last_detection = None
        self.last_error = None
        self.events_emitted = 0
        self.analysis_iterations = 0
        self.onset_count = 0
        self.last_onset_at = None
        self.last_analysis_ms = None
        self.last_audio_metrics = None
        self.last_result_snapshot = None
        self._last_diag_log_at = 0.0
        self._resume_logged = True

    def _iter_input_device_candidates(self):
        resolved_device, resolved_info = resolve_audio_input_device(self.input_device_preference)
        self.input_device_info = resolved_info or {}

        candidates = []
        for device in (self.input_device_preference, resolved_device):
            if device is None:
                continue
            if any(device == existing for existing in candidates):
                continue
            candidates.append(device)
        return candidates

    def _publish(self, payload):
        event = {
            "source": "audio",
            "timestamp": time.time(),
        }
        event.update(payload)
        self.event_callback(event)
        self.last_publish_at = event["timestamp"]
        self.events_emitted += 1
        self.last_detection = {
            "species": event.get("species"),
            "confidence": event.get("confidence"),
            "emotion": event.get("emotion"),
            "emotion_similarity": event.get("emotion_similarity"),
            "doa_angle": event.get("doa_angle"),
            "target_angle": event.get("target_angle"),
            "amplitude": event.get("amplitude"),
            "probabilities": event.get("probabilities"),
            "timestamp": event["timestamp"],
        }

    def get_status(self):
        return {
            "alive": self.is_alive(),
            "started_at": self.started_at,
            "stream_started_at": self.stream_started_at,
            "last_loop_at": self.last_loop_at,
            "last_publish_at": self.last_publish_at,
            "events_emitted": self.events_emitted,
            "analysis_iterations": self.analysis_iterations,
            "onset_count": self.onset_count,
            "last_onset_at": self.last_onset_at,
            "last_analysis_ms": self.last_analysis_ms,
            "last_audio_metrics": self.last_audio_metrics,
            "last_result_snapshot": self.last_result_snapshot,
            "spectrum_updated_at": self._spectrum_updated_at,
            "last_detection": self.last_detection,
            "last_error": self.last_error,
            "input_device": self.input_device,
            "input_device_preference": self.input_device_preference,
            "input_device_info": self.input_device_info,
        }

    def _device_label(self):
        info = self.input_device_info or {}
        device_name = info.get("name")
        if device_name:
            return f"{self.input_device} ({device_name})"
        return str(self.input_device)

    def _format_top_probs(self, probs, top_k=3):
        if probs is None:
            return "none"
        try:
            arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        except Exception:
            return "invalid"
        if arr.size == 0:
            return "empty"
        top_idx = np.argsort(arr)[::-1][:top_k]
        parts = []
        for idx in top_idx:
            if idx < len(audio_core.CLASS_NAMES):
                name = audio_core.CLASS_NAMES[idx]
            else:
                name = f"class_{idx}"
            parts.append(f"{name}:{float(arr[idx]):.2f}")
        return ", ".join(parts)

    def _log_diag(self, message, force=False):
        now = time.time()
        if force or now - self._last_diag_log_at >= self.DIAG_LOG_INTERVAL_S:
            print(f"[audio][diag] {message}", flush=True)
            self._last_diag_log_at = now

    def _audio_cb(self, indata, frames, time_info, status):
        del frames, time_info, status
        samples = indata[:, 0].astype(np.float32)
        count = len(samples)
        is_loud = float(np.max(np.abs(samples))) > 0.01
        if self._prev_quiet and is_loud:
            with self._audio_lock:
                self._audio_buf[:] = 0
            self._skip_analyze = 10
            self._resume_logged = False
            self.onset_count += 1
            self.last_onset_at = time.time()
            self._log_diag(
                f"onset#{self.onset_count} amp={float(np.max(np.abs(samples))):.3f} "
                f"skip={self._skip_analyze} device={self._device_label()}",
                force=True,
            )
        self._prev_quiet = not is_loud
        with self._audio_lock:
            self._audio_buf[:-count] = self._audio_buf[count:]
            self._audio_buf[-count:] = samples
            # Keep a same-source chunk for the dashboard spectrum so we do not
            # open a second audio stream and disturb recognition timing.
            self._latest_chunk = samples.copy()
            self._latest_chunk_seq += 1

    def _compute_spectrum_bins(self, samples):
        if samples is None or samples.size == 0:
            return [0.0] * self.SPECTRUM_BINS
        if not np.any(samples):
            return [0.0] * self.SPECTRUM_BINS

        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < 1e-4:
            return [0.0] * self.SPECTRUM_BINS

        window = np.hanning(samples.size)
        windowed = samples * window
        fft = np.fft.rfft(windowed)
        mag = np.abs(fft) / max(float(window.sum()), 1.0)
        if mag.size <= 1:
            return [0.0] * self.SPECTRUM_BINS

        # Drop DC and map the chunk into a fixed dB range so quiet input stays quiet
        # on the dashboard instead of being stretched to full scale every frame.
        mag = mag[1:]
        groups = np.array_split(mag, self.SPECTRUM_BINS)
        bins = np.array(
            [float(np.sqrt(np.mean(group ** 2))) if group.size else 0.0 for group in groups],
            dtype=np.float32,
        )
        bins = 20.0 * np.log10(bins + 1e-6)
        bins = np.clip(bins, self.SPECTRUM_MIN_DB, self.SPECTRUM_MAX_DB)
        bins = (bins - self.SPECTRUM_MIN_DB) / (self.SPECTRUM_MAX_DB - self.SPECTRUM_MIN_DB)
        return bins.tolist()

    def get_spectrum_snapshot(self):
        with self._audio_lock:
            chunk = self._latest_chunk.copy() if self._latest_chunk is not None else None
            seq = self._latest_chunk_seq
            cache_seq = self._spectrum_cache_seq
            cached_bins = list(self._spectrum_bins)
            updated_at = self._spectrum_updated_at

        if seq == cache_seq:
            return {"bins": cached_bins, "updated_at": updated_at, "seq": seq}

        bins = np.array(self._compute_spectrum_bins(chunk), dtype=np.float32)
        now = time.time()
        with self._audio_lock:
            if seq >= self._spectrum_cache_seq:
                self._spectrum_smoothed = (
                    self._spectrum_smoothed * self.SPECTRUM_SMOOTHING
                    + bins * (1.0 - self.SPECTRUM_SMOOTHING)
                )
                self._spectrum_bins = self._spectrum_smoothed.tolist()
                self._spectrum_cache_seq = seq
                self._spectrum_updated_at = now
            return {
                "bins": list(self._spectrum_bins),
                "updated_at": self._spectrum_updated_at,
                "seq": self._spectrum_cache_seq,
            }

    def _normalize_result(self, result, current_amp):
        species = result["species"]
        conf = result["conf"]
        probs = result.get("probs")

        if probs is not None:
            if probs[2] > 0.30:
                species = "snake"
                conf = probs[2]
            elif probs[3] > 0.60:
                species = "weasel"
                conf = probs[3]

        if conf < 0.10:
            return None
        if species == "background":
            return None
        if current_amp < 0.15:
            return None
        if species == "gun" and conf < 0.50:
            return None

        return species, conf

    def _get_doa(self, hw, species):
        if hw.dev is not None:
            return hw.get_angle()
        return audio_core.SPECIES_ANGLE_MAP.get(species, 90)

    def _smooth_doa(self, angle):
        if self._doa_smooth is None:
            self._doa_smooth = float(angle)
            self._doa_angle_prev = float(angle)
            return self._doa_smooth

        diff = angle - self._doa_angle_prev
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        self._doa_smooth = (self._doa_smooth + diff * audio_core._DOA_EMA_ALPHA) % 360
        if self._doa_smooth < 0:
            self._doa_smooth += 360
        self._doa_angle_prev = float(angle)
        return self._doa_smooth

    def _build_event(self, result, species, conf, doa_angle, smoothed_angle, current_amp):
        target_angle = doa_angle
        if target_angle > 180:
            target_angle -= 360
        target_angle = max(-90, min(90, target_angle))
        return {
            "type": "audio_detection",
            "species": species,
            "confidence": float(conf),
            "priority": int(result["prio"]),
            "emotion": result.get("emotion"),
            "emotion_similarity": float(result.get("emotion_sim", 0.0)),
            "doa_angle": float(doa_angle),
            "smoothed_angle": float(smoothed_angle),
            "target_angle": float(target_angle),
            "amplitude": float(current_amp),
            "probabilities": [float(p) for p in result.get("probs", [])],
        }

    def run(self):
        self.started_at = time.time()
        try:
            hw = audio_core.ReSpeakerXVF3800()
            engine = audio_core.IntegratedEngine()
            candidates = self._iter_input_device_candidates()
            last_exc = None
            for candidate in candidates:
                try:
                    print(f"[audio] opening InputStream device={candidate} rate={audio_core.RATE}", flush=True)
                    self._audio_stream = sd.InputStream(
                        device=candidate,
                        samplerate=audio_core.RATE,
                        channels=2,
                        dtype="float32",
                        blocksize=int(audio_core.RATE * 0.1),
                        callback=self._audio_cb,
                    )
                    self.input_device = candidate
                    break
                except Exception as exc:
                    last_exc = exc
                    self._audio_stream = None
                    print(f"[audio] open failed on device={candidate}: {exc}", flush=True)
            if self._audio_stream is None:
                raise RuntimeError(f"no usable audio input device; candidates={candidates}, last_error={last_exc}")
            self._audio_stream.start()
            self.stream_started_at = time.time()
            print("[audio] stream started", flush=True)
            self._log_diag(f"stream ready device={self._device_label()}", force=True)

            time.sleep(2.0)
            self._skip_analyze = 0
            while not self.stop_event.is_set():
                self.last_loop_at = time.time()
                if self._skip_analyze > 0:
                    self._skip_analyze -= 1
                    time.sleep(0.2)
                    continue
                if not self._resume_logged:
                    self._resume_logged = True
                    self._log_diag("analysis resumed after onset buffer refill", force=True)

                time.sleep(0.2)
                with self._audio_lock:
                    audio_data = self._audio_buf.copy()

                current_amp = np.max(np.abs(audio_data))
                if current_amp < audio_core.SILENCE_THRESHOLD:
                    continue

                rms = float(np.sqrt(np.mean(audio_data ** 2)) + 1e-9)
                if rms < 0.01:
                    audio_data = audio_data / rms * 0.05
                else:
                    audio_data = audio_data / rms * 0.1

                self.analysis_iterations += 1
                analyze_started_at = time.time()
                result = engine.analyze(audio_data)
                self.last_analysis_ms = round((time.time() - analyze_started_at) * 1000.0, 2)
                self.last_audio_metrics = {
                    "amp": round(float(current_amp), 4),
                    "rms": round(rms, 4),
                    "skip_analyze": int(self._skip_analyze),
                    "device": self._device_label(),
                    "analysis_iteration": int(self.analysis_iterations),
                }
                if not result:
                    self.last_result_snapshot = {
                        "species": None,
                        "confidence": None,
                        "top_probs": "none",
                    }
                    continue

                top_probs = self._format_top_probs(result.get("probs"))
                self.last_result_snapshot = {
                    "species": result.get("species"),
                    "confidence": None if result.get("conf") is None else round(float(result["conf"]), 4),
                    "top_probs": top_probs,
                }
                self._log_diag(
                    f"iter={self.analysis_iterations} ms={self.last_analysis_ms} "
                    f"amp={current_amp:.3f} rms={rms:.3f} raw={result.get('species')} "
                    f"conf={float(result.get('conf', 0.0)):.2f} top=[{top_probs}]"
                )

                normalized = self._normalize_result(result, current_amp)
                if normalized is None:
                    continue
                species, conf = normalized

                doa_angle = self._get_doa(hw, species)
                smoothed_angle = self._smooth_doa(doa_angle)
                should_emit = abs(smoothed_angle - self._last_angle) > audio_core.ANGLE_THRESHOLD or result["prio"] <= 4
                if not should_emit:
                    continue

                now = time.time()
                last_seen = self._last_species_time.get(species, 0)
                if now - last_seen < 10.0:
                    continue
                self._last_species_time[species] = now
                self._last_angle = smoothed_angle
                self._log_diag(
                    f"publish species={species} conf={conf:.2f} doa={float(doa_angle):.1f} "
                    f"target={float(max(-90, min(90, doa_angle if doa_angle <= 180 else doa_angle - 360))):.1f} "
                    f"amp={current_amp:.3f}",
                    force=True,
                )
                self._publish(self._build_event(result, species, conf, doa_angle, smoothed_angle, current_amp))
        except Exception as exc:
            self.last_error = repr(exc)
            print(f"[audio] worker failed: {exc}", flush=True)
            raise
        finally:
            if self._audio_stream is not None:
                try:
                    self._audio_stream.stop()
                    self._audio_stream.close()
                except Exception:
                    pass
