from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace

from ..control.trajectories import (
    ChirpTrajectory,
    ConstantTrajectory,
    ConstantSpeedTrajectory,
    FreeDecayTrajectory,
    StepTrajectory,
)


@dataclass(frozen=True)
class ExperimentSpec:
    mode: str
    parameters: dict = field(default_factory=dict)
    kp: float = 0.0
    kd: float = 0.0
    tau_ff: float = 0.0
    notes: str = ""


class ExperimentManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._trajectory = ConstantTrajectory(0.0)
        self._spec = ExperimentSpec("manual")
        self._started = time.perf_counter()
        self._id = "manual"
        self._active = False
        self._last_event = ""

    def _make_trajectory(self, spec):
        p = spec.parameters
        if spec.mode in {"manual", "hold"}:
            if spec.mode == "manual" and "initial_position" in p:
                return ConstantSpeedTrajectory(
                    p["initial_position"], p["position"], p["speed_rad_s"]
                )
            return ConstantTrajectory(p.get("position", 0.0))
        if spec.mode == "step":
            return StepTrajectory(
                p["initial_position"], p["step_amplitude"],
                p["pre_hold_s"], p["post_duration_s"],
            )
        if spec.mode == "chirp":
            return ChirpTrajectory(
                p["center_position"], p["amplitude"], p["f_start_hz"],
                p["f_end_hz"], p["duration_s"], p.get("kind", "linear"),
            )
        if spec.mode == "free_decay":
            return FreeDecayTrajectory(
                p["duration_s"], p.get("disabled", True), p.get("hold_position", 0.0)
            )
        raise ValueError(f"unknown experiment mode: {spec.mode}")

    def start(self, spec, now=None):
        trajectory = self._make_trajectory(spec)
        with self._lock:
            self._spec = spec
            self._trajectory = trajectory
            self._started = time.perf_counter() if now is None else float(now)
            self._id = f"{spec.mode}-{uuid.uuid4().hex[:10]}"
            self._active = True
            self._last_event = "experiment_start"
            return self._id

    def set_manual_target(self, position, kp, kd, tau_ff=0.0, *, initial_position=None, speed_rad_s=None):
        parameters = {"position": float(position)}
        if initial_position is not None:
            parameters.update({
                "initial_position": float(initial_position),
                "speed_rad_s": float(speed_rad_s),
            })
        return self.start(ExperimentSpec(
            mode="manual", parameters=parameters,
            kp=float(kp), kd=float(kd), tau_ff=float(tau_ff),
        ))

    def stop(self):
        with self._lock:
            self._active = False
            self._last_event = "experiment_stop"

    def update_gains(self, kp, kd, tau_ff=None):
        """Update impedance fields without resetting trajectory phase/time."""
        with self._lock:
            self._spec = replace(
                self._spec,
                kp=float(kp),
                kd=float(kd),
                tau_ff=self._spec.tau_ff if tau_ff is None else float(tau_ff),
            )
            self._last_event = "gain_update"

    def sample(self, now):
        with self._lock:
            elapsed = max(0.0, float(now) - self._started)
            point = self._trajectory.sample(elapsed)
            event = self._last_event or point.event
            self._last_event = ""
            if point.complete:
                self._active = False
            return self._spec, self._id, elapsed, self._active, point, event

    def snapshot(self):
        with self._lock:
            return self._spec, self._id, self._active
