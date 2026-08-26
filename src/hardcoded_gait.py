#!/usr/bin/env python3
"""Periodic joint-space gait replay fitted from validated simulation logs."""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import yaml

from joint_mapping import POLICY_JOINT_ORDER


@dataclass(frozen=True)
class FourierGaitTemplate:
    name: str
    source_frequency_hz: float
    coefficients: np.ndarray

    def __post_init__(self):
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.shape != (7, len(POLICY_JOINT_ORDER)):
            raise ValueError(
                f"{self.name}: coefficients must have shape (7, 12), got "
                f"{coefficients.shape}"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError(f"{self.name}: coefficients contain non-finite values")
        if not np.isfinite(self.source_frequency_hz) or self.source_frequency_hz <= 0.0:
            raise ValueError(f"{self.name}: source_frequency_hz must be finite and > 0")
        object.__setattr__(self, "coefficients", coefficients)

    def sample(self, phase_cycles):
        phase = 2.0 * math.pi * float(phase_cycles)
        result = self.coefficients[0].copy()
        for harmonic in range(1, 4):
            result += self.coefficients[2 * harmonic - 1] * math.sin(harmonic * phase)
            result += self.coefficients[2 * harmonic] * math.cos(harmonic * phase)
        return result.astype(np.float32)


class HardcodedGaitPlayer:
    """Generate smooth forward/backward targets without running the actor."""

    def __init__(
        self,
        templates,
        amplitude_scale=0.5,
        frequency_scale=0.5,
        direction_blend_seconds=2.0,
        max_target_step_rad=0.025,
    ):
        self.templates = dict(templates)
        self.amplitude_scale = self._bounded_scale(amplitude_scale, "amplitude_scale")
        self.frequency_scale = self._bounded_scale(frequency_scale, "frequency_scale")
        self.direction_blend_seconds = max(0.0, float(direction_blend_seconds))
        self.max_target_step_rad = float(max_target_step_rad)
        if not np.isfinite(self.max_target_step_rad) or self.max_target_step_rad <= 0.0:
            raise ValueError("max_target_step_rad must be finite and > 0")
        self.reset()

    @staticmethod
    def _bounded_scale(value, label):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError(f"{label} must be in (0, 1]")
        return value

    @classmethod
    def from_yaml(cls, path, amplitude_scale=None, frequency_scale=None):
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}
        if list(cfg.get("joint_order", [])) != list(POLICY_JOINT_ORDER):
            raise ValueError("hardcoded gait joint_order does not match policy order")
        templates = {}
        for name in ("forward", "backward"):
            item = (cfg.get("templates", {}) or {}).get(name, {}) or {}
            templates[name] = FourierGaitTemplate(
                name=name,
                source_frequency_hz=float(item["source_frequency_hz"]),
                coefficients=np.asarray(item["coefficients"], dtype=np.float64),
            )
        defaults = cfg.get("deployment_defaults", {}) or {}
        return cls(
            templates,
            amplitude_scale=(
                defaults.get("amplitude_scale", 0.5)
                if amplitude_scale is None
                else amplitude_scale
            ),
            frequency_scale=(
                defaults.get("frequency_scale", 0.5)
                if frequency_scale is None
                else frequency_scale
            ),
            direction_blend_seconds=float(defaults.get("direction_blend_seconds", 2.0)),
            max_target_step_rad=float(defaults.get("max_target_step_rad", 0.025)),
        )

    def reset(self):
        self.direction = None
        self.phase_cycles = 0.0
        self.last_target = np.zeros(len(POLICY_JOINT_ORDER), dtype=np.float32)
        self._blend_start = self.last_target.copy()
        self._blend_elapsed = 0.0

    def update(self, direction, dt, current_target=None):
        direction = "forward" if float(direction) > 0.0 else "backward"
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0")
        if direction != self.direction:
            self.direction = direction
            self.phase_cycles = 0.0
            self._blend_elapsed = 0.0
            self._blend_start = np.asarray(
                self.last_target if current_target is None else current_target,
                dtype=np.float32,
            ).copy()
            self.last_target = self._blend_start.copy()
        template = self.templates[direction]
        self.phase_cycles = (
            self.phase_cycles
            + dt * template.source_frequency_hz * self.frequency_scale
        ) % 1.0
        target = self.amplitude_scale * template.sample(self.phase_cycles)
        if self.direction_blend_seconds > 0.0:
            self._blend_elapsed = min(
                self.direction_blend_seconds,
                self._blend_elapsed + dt,
            )
            x = self._blend_elapsed / self.direction_blend_seconds
            alpha = x * x * (3.0 - 2.0 * x)
            target = (1.0 - alpha) * self._blend_start + alpha * target
        target = self.last_target + np.clip(
            np.asarray(target, dtype=np.float32) - self.last_target,
            -self.max_target_step_rad,
            self.max_target_step_rad,
        )
        self.last_target = np.asarray(target, dtype=np.float32)
        return self.last_target.copy()
