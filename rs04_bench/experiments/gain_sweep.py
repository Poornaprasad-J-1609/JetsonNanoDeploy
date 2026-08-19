from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..analysis.step_response import analyze_step_response
from .manager import ExperimentSpec


@dataclass(frozen=True)
class SweepResult:
    kp: float
    kd: float
    cost: float
    metrics: dict
    csv_path: str
    safe: bool
    note: str = ""


class GainSweepRunner:
    """Sequential small-step sweep with whole-sweep abort on any safety event."""

    def __init__(self, controller, callback=None):
        self.controller = controller
        self.callback = callback
        self.results = []
        self.error = ""
        self._abort = threading.Event()
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def abort(self):
        self._abort.set()

    def start(
        self, kp_values, kd_values, initial_position, step_amplitude,
        pre_hold_s=1.0, post_duration_s=3.0, return_settle_s=1.5,
        weights=None, saturation_fraction=0.95,
    ):
        if self.running:
            raise RuntimeError("gain sweep is already running")
        self.results = []
        self.error = ""
        self._abort.clear()
        args = (list(kp_values), list(kd_values), float(initial_position), float(step_amplitude),
                float(pre_hold_s), float(post_duration_s), float(return_settle_s),
                weights or {}, float(saturation_fraction))
        self._thread = threading.Thread(target=self._run, args=args, name="rs04-gain-sweep", daemon=True)
        self._thread.start()

    def _wait_for(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._abort.is_set():
            snapshot = self.controller.snapshot()
            if snapshot.safety_event:
                raise RuntimeError(snapshot.safety_event)
            if predicate(snapshot):
                return snapshot
            time.sleep(0.02)
        raise RuntimeError("gain-sweep stage timed out")

    def _run(self, kp_values, kd_values, initial, amplitude, pre_hold, duration, settle, weights, sat_fraction):
        weights = {"rms": 1.0, "settling": 0.2, "overshoot": 0.02, "current": 0.05, "saturation": 10.0, **weights}
        try:
            for kp in kp_values:
                for kd in kd_values:
                    if self._abort.is_set():
                        raise RuntimeError("gain sweep aborted by operator")
                    self.controller.set_manual_target(initial)
                    self._wait_for(
                        lambda s: s.state is not None and abs(s.state.position - initial) < 0.02 and abs(s.state.velocity) < 0.05,
                        settle + 3.0,
                    )
                    spec = ExperimentSpec(
                        mode="step", kp=float(kp), kd=float(kd),
                        parameters={
                            "initial_position": initial, "step_amplitude": amplitude,
                            "pre_hold_s": pre_hold, "post_duration_s": duration,
                        },
                        notes="automated safe gain sweep",
                    )
                    self.controller.start_experiment(spec)
                    self._wait_for(lambda s: not s.experiment_active, pre_hold + duration + 2.0)
                    self._wait_for(lambda s: not self.controller.logger.active, 4.0)
                    path = self.controller.snapshot().last_csv_path
                    metrics = analyze_step_response(path)
                    peak_torque = metrics.get("maximum_measured_torque_nm") or metrics.get("maximum_commanded_torque_nm") or 0.0
                    saturated = peak_torque >= sat_fraction * self.controller.config.safety.max_torque_nm
                    current = metrics.get("maximum_current_a") or 0.0
                    cost = (
                        weights["rms"] * metrics["rms_position_error_rad"]
                        + weights["settling"] * (metrics["settling_time_s"] or duration)
                        + weights["overshoot"] * metrics["overshoot_percent"]
                        + weights["current"] * current
                        + weights["saturation"] * int(saturated)
                    )
                    result = SweepResult(kp, kd, cost, metrics, path, not saturated,
                                         "torque saturation penalty" if saturated else "")
                    self.results.append(result)
                    if self.callback:
                        self.callback(result)
                    if saturated:
                        raise RuntimeError("torque saturation threshold reached; entire sweep aborted")
                    self.controller.set_manual_target(initial)
                    time.sleep(settle)
        except Exception as exc:
            self.error = str(exc)
            self.controller.emergency_stop(f"gain sweep aborted: {exc}")

    def best_candidates(self, count=5):
        return sorted((item for item in self.results if item.safe), key=lambda item: item.cost)[:count]
