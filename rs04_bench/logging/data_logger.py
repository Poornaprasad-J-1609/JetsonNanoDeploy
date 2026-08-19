from __future__ import annotations

import csv
import json
import queue
import subprocess
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path


SAMPLE_FIELDS = [
    "timestamp_wall", "timestamp_monotonic", "experiment_time",
    "control_dt", "control_frequency", "experiment_mode",
    "q_target_requested_rad", "q_des_rad", "q_actual_rad", "position_error_rad",
    "qd_des_rad_s", "qd_actual_rad_s", "velocity_error_rad_s",
    "kp", "kd", "tau_ff_nm", "tau_commanded_nm", "tau_measured_nm",
    "tau_estimated_nm", "motor_current_a", "motor_voltage_v",
    "motor_temperature_c", "motor_enabled", "safety_event", "experiment_event",
    "experiment_id", "cycle_index", "deadline_lateness_s", "missed_cycles_total",
    "feedback_age_s", "motor_fault_bits", "motor_mode_status",
]


def _jsonable(value):
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def git_commit(root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


class AsyncDataLogger:
    def __init__(self, directory, queue_size=20_000, flush_interval_s=0.5):
        self.directory = Path(directory).expanduser().resolve()
        self.queue = queue.Queue(maxsize=int(queue_size))
        self.flush_interval_s = float(flush_interval_s)
        self.thread = None
        self.csv_path = None
        self.metadata_path = None
        self._stop = threading.Event()
        self._error = None
        self._dropped = 0
        self._rows = 0
        self._lifecycle_lock = threading.Lock()

    @property
    def active(self):
        # Remain active through metadata finalization, not just until the CSV
        # writer thread exits. This prevents a new experiment from reusing the
        # logger while the previous JSON is still being completed.
        return self.thread is not None

    @property
    def error(self):
        return self._error

    @property
    def dropped_samples(self):
        return self._dropped

    def start(self, experiment_mode, metadata):
        if self.active:
            raise RuntimeError("a logger session is already active")
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stem = f"rs04_{experiment_mode}_{stamp}"
        self.csv_path = self.directory / f"{stem}.csv"
        self.metadata_path = self.directory / f"{stem}_metadata.json"
        meta = dict(metadata)
        meta.update({
            "created_at": datetime.now().astimezone().isoformat(),
            "csv_path": str(self.csv_path),
            "software_git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "availability": {
                "blank_values": "Signal was not exposed by the active hardware/API",
                "torque_measured": metadata.get("torque_feedback_source", "unknown"),
                "current": metadata.get("current_source", "unknown"),
                "voltage": metadata.get("voltage_source", "unknown"),
            },
        })
        self.metadata_path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
        self._stop.clear()
        self._error = None
        self._dropped = 0
        self._rows = 0
        self.thread = threading.Thread(target=self._writer, name="rs04-csv-writer", daemon=True)
        self.thread.start()
        return self.csv_path, self.metadata_path

    def log(self, row):
        if not self.active:
            return
        try:
            self.queue.put_nowait(dict(row))
        except queue.Full:
            self._dropped += 1
            raise RuntimeError("CSV logging queue is full; experiment data would be incomplete")

    def _writer(self):
        try:
            with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
                writer.writeheader()
                last_flush = time.monotonic()
                while not self._stop.is_set() or not self.queue.empty():
                    try:
                        row = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        row = None
                    if row is not None:
                        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in SAMPLE_FIELDS})
                        self._rows += 1
                    now = time.monotonic()
                    if now - last_flush >= self.flush_interval_s:
                        stream.flush()
                        last_flush = now
                stream.flush()
        except Exception as exc:
            self._error = exc

    def stop(self, extra_metadata=None):
        with self._lifecycle_lock:
            thread = self.thread
            if thread is None:
                return self.csv_path
            self._stop.set()
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("CSV writer did not stop")
            if self.metadata_path and self.metadata_path.exists():
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                metadata.update({
                    "completed_at": datetime.now().astimezone().isoformat(),
                    "sample_count": self._rows,
                    "dropped_samples": self._dropped,
                    "writer_error": "" if self._error is None else str(self._error),
                })
                if extra_metadata:
                    metadata.update(_jsonable(extra_metadata))
                self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.thread = None
            return self.csv_path
