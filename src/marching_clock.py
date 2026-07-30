#!/usr/bin/env python3
"""Marching-clock generation with an explicit diagnostic-only boundary."""

from __future__ import annotations

import math
import time

import numpy as np


DIAGNOSTIC_CLOCK_FORMULAS = (
    "sin-cos-zero",
    "sin-cos-phase",
    "three-phase-sine",
)


class MarchingClockConfigurationError(ValueError):
    pass


class MarchingClock:
    """Generate the three observation slots used by model_12357.

    A diagnostic clock is intentionally marked as such because selecting a
    plausible periodic encoding is not equivalent to recovering the exact
    Isaac training contract.
    """

    def __init__(
        self,
        formula: str,
        frequency_hz: float,
        *,
        diagnostic_only: bool,
        reset_rule: str,
        time_source=time.monotonic,
    ):
        formula = str(formula).strip().lower()
        if formula not in DIAGNOSTIC_CLOCK_FORMULAS:
            raise MarchingClockConfigurationError(
                f"unsupported marching-clock formula {formula!r}; "
                f"choose one of {DIAGNOSTIC_CLOCK_FORMULAS}"
            )
        try:
            frequency_hz = float(frequency_hz)
        except (TypeError, ValueError) as exc:
            raise MarchingClockConfigurationError(
                "marching-clock frequency must be finite and > 0"
            ) from exc
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise MarchingClockConfigurationError(
                "marching-clock frequency must be finite and > 0"
            )
        self.formula = formula
        self.frequency_hz = frequency_hz
        self.diagnostic_only = bool(diagnostic_only)
        self.reset_rule = str(reset_rule)
        self._time_source = time_source
        self._reset_time = float(self._time_source())

    @classmethod
    def unverified_shadow(cls, formula: str, frequency_hz: float):
        return cls(
            formula,
            frequency_hz,
            diagnostic_only=True,
            reset_rule="shadow_loop_start",
        )

    @classmethod
    def from_verified_contract(cls, observation_contract: dict):
        if not bool(observation_contract.get("clock_formula_verified", False)):
            raise MarchingClockConfigurationError(
                "policy contract does not verify the marching-clock formula"
            )
        if not bool(observation_contract.get("clock_provider_implemented", False)):
            raise MarchingClockConfigurationError(
                "policy contract does not approve the marching-clock provider"
            )
        reset_rule = str(observation_contract.get("clock_reset_rule") or "")
        if reset_rule != "policy_loop_start":
            raise MarchingClockConfigurationError(
                "only the verified reset rule 'policy_loop_start' is implemented; "
                f"contract contains {reset_rule!r}"
            )
        return cls(
            observation_contract.get("clock_formula"),
            observation_contract.get("clock_frequency_hz"),
            diagnostic_only=False,
            reset_rule=reset_rule,
        )

    def reset(self, now: float | None = None):
        self._reset_time = (
            float(self._time_source()) if now is None else float(now)
        )

    def sample_elapsed(self, elapsed_s: float) -> np.ndarray:
        elapsed_s = float(elapsed_s)
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("marching-clock elapsed time must be finite and >= 0")
        turns = (elapsed_s * self.frequency_hz) % 1.0
        phase = 2.0 * math.pi * turns
        sin_phase = math.sin(phase)
        cos_phase = math.cos(phase)
        if self.formula == "sin-cos-zero":
            values = (sin_phase, cos_phase, 0.0)
        elif self.formula == "sin-cos-phase":
            values = (sin_phase, cos_phase, turns)
        else:
            values = (
                sin_phase,
                math.sin(phase + 2.0 * math.pi / 3.0),
                math.sin(phase + 4.0 * math.pi / 3.0),
            )
        return np.asarray(values, dtype=np.float32)

    def sample(self, now: float | None = None) -> np.ndarray:
        current = float(self._time_source()) if now is None else float(now)
        return self.sample_elapsed(max(0.0, current - self._reset_time))

    def require_runtime_mode(self, *, policy_shadow_mode: bool, mode: str):
        if self.diagnostic_only and (
            not bool(policy_shadow_mode) or str(mode) != "print"
        ):
            raise MarchingClockConfigurationError(
                "an unverified marching clock is permitted only with "
                "--policy-shadow-mode --mode print; it can never drive motors"
            )
