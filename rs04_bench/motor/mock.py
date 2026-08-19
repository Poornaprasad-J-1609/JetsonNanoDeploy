from __future__ import annotations

import math
import threading
import time
from collections import deque

from .interface import MotorCommand, MotorInterface, MotorState


class MockMotorInterface(MotorInterface):
    """Deterministic second-order actuator with delay and friction."""

    def __init__(self, config):
        self.config = config
        self._connected = False
        self._enabled = False
        self._q = 0.0
        self._qd = 0.0
        self._temperature = float(config.initial_temperature_c)
        self._last_update = time.perf_counter()
        self._commands = deque()
        self._active = MotorCommand()
        self._lock = threading.Lock()

    @property
    def connected(self):
        return self._connected

    @property
    def enabled(self):
        return self._enabled

    def connect(self):
        self._connected = True
        self._last_update = time.perf_counter()

    def disconnect(self):
        self.disable()
        self._connected = False

    def enable(self):
        if not self._connected:
            raise RuntimeError("mock motor is not connected")
        self._enabled = True

    def disable(self):
        self._enabled = False
        self._active = MotorCommand(q_des=self._q)
        self._commands.clear()

    def send_command(self, command):
        if not self._connected:
            raise RuntimeError("mock motor is not connected")
        with self._lock:
            self._commands.append((time.perf_counter() + self.config.command_delay_s, command))

    def _advance(self, now):
        dt_total = max(0.0, min(0.05, now - self._last_update))
        self._last_update = now
        with self._lock:
            while self._commands and self._commands[0][0] <= now:
                _, self._active = self._commands.popleft()
        substeps = max(1, int(math.ceil(dt_total / 0.001)))
        dt = dt_total / substeps
        tau = 0.0
        for _ in range(substeps):
            if self._enabled:
                command = self._active
                tau = (
                    command.kp * (command.q_des - self._q)
                    + command.kd * (command.qd_des - self._qd)
                    + command.tau_ff
                )
                tau = max(-self.config.torque_limit_nm, min(self.config.torque_limit_nm, tau))
            else:
                tau = 0.0
            friction = self.config.coulomb_friction_nm * math.tanh(self._qd / 0.01)
            qdd = (tau - self.config.viscous_damping_nm_s_rad * self._qd - friction) / self.config.inertia_kg_m2
            self._qd += qdd * dt
            self._q += self._qd * dt
            self._temperature += (0.0004 * tau * tau - 0.02 * (self._temperature - 25.0)) * dt
        return tau

    def read_state(self, timeout_s=0.0):
        if not self._connected:
            return None
        now = time.perf_counter()
        tau = self._advance(now)
        return MotorState(
            timestamp_monotonic=now,
            position=self._q,
            velocity=self._qd,
            torque_measured=tau,
            current=None,
            temperature=self._temperature,
            voltage=None,
            fault_bits=0,
            mode_status=2 if self._enabled else 0,
            feedback_source="mock_exact_torque",
        )
