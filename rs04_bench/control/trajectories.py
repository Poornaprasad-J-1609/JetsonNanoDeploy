from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryPoint:
    q: float
    qd: float
    complete: bool = False
    event: str = ""


class ConstantTrajectory:
    def __init__(self, position):
        self.position = float(position)

    def sample(self, elapsed):
        return TrajectoryPoint(self.position, 0.0)


class ConstantSpeedTrajectory:
    """Point-to-point motion with a constant signed velocity and no overshoot."""

    def __init__(self, initial, target, speed):
        self.initial = float(initial)
        self.target = float(target)
        self.speed = float(speed)
        if not math.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("manual speed must be a positive finite value")
        self.distance = self.target - self.initial
        self.direction = 0.0 if math.isclose(self.distance, 0.0, abs_tol=1e-12) else math.copysign(1.0, self.distance)
        self.duration = abs(self.distance) / self.speed

    def sample(self, elapsed):
        elapsed = max(0.0, float(elapsed))
        if self.direction == 0.0 or elapsed >= self.duration:
            return TrajectoryPoint(self.target, 0.0)
        return TrajectoryPoint(
            self.initial + self.direction * self.speed * elapsed,
            self.direction * self.speed,
        )


class StepTrajectory:
    def __init__(self, initial, amplitude, pre_hold_s, post_duration_s):
        self.initial = float(initial)
        self.amplitude = float(amplitude)
        self.pre_hold_s = float(pre_hold_s)
        self.post_duration_s = float(post_duration_s)

    def sample(self, elapsed):
        stepped = elapsed >= self.pre_hold_s
        complete = elapsed >= self.pre_hold_s + self.post_duration_s
        return TrajectoryPoint(
            self.initial + (self.amplitude if stepped else 0.0),
            0.0,
            complete=complete,
            event="step" if stepped and elapsed < self.pre_hold_s + 0.005 else "",
        )


class ChirpTrajectory:
    def __init__(self, center, amplitude, f_start, f_end, duration, kind="linear"):
        self.center = float(center)
        self.amplitude = float(amplitude)
        self.f_start = float(f_start)
        self.f_end = float(f_end)
        self.duration = float(duration)
        self.kind = str(kind)
        if self.duration <= 0 or self.f_start <= 0 or self.f_end <= 0:
            raise ValueError("chirp duration and frequencies must be positive")
        if self.kind not in {"linear", "logarithmic"}:
            raise ValueError("chirp kind must be linear or logarithmic")

    def _phase_frequency(self, t):
        t = min(max(0.0, t), self.duration)
        if self.kind == "linear":
            rate = (self.f_end - self.f_start) / self.duration
            phase = 2.0 * math.pi * (self.f_start * t + 0.5 * rate * t * t)
            frequency = self.f_start + rate * t
        elif math.isclose(self.f_start, self.f_end):
            phase = 2.0 * math.pi * self.f_start * t
            frequency = self.f_start
        else:
            ratio = self.f_end / self.f_start
            log_ratio = math.log(ratio)
            phase = 2.0 * math.pi * self.f_start * self.duration * (
                math.exp(log_ratio * t / self.duration) - 1.0
            ) / log_ratio
            frequency = self.f_start * math.exp(log_ratio * t / self.duration)
        return phase, frequency

    def sample(self, elapsed):
        phase, frequency = self._phase_frequency(elapsed)
        q = self.center + self.amplitude * math.sin(phase)
        qd = self.amplitude * math.cos(phase) * 2.0 * math.pi * frequency
        return TrajectoryPoint(q, qd, complete=elapsed >= self.duration)


class FreeDecayTrajectory:
    """Recording trajectory; motor command policy is selected explicitly."""

    def __init__(self, duration, disabled=True, hold_position=0.0):
        self.duration = float(duration)
        self.disabled = bool(disabled)
        self.hold_position = float(hold_position)

    def sample(self, elapsed):
        return TrajectoryPoint(self.hold_position, 0.0, elapsed >= self.duration)
