from __future__ import annotations

import sys
import time
from pathlib import Path

from .interface import MotorCommand, MotorInterface, MotorState


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motor_command_layer import (  # noqa: E402
    decode_mit_feedback_frame,
    mit_can_id,
    motor_management_can_id,
    pack_mit_command,
)
from robstride_can_interface import SocketCan  # noqa: E402


def official_rs04_protocol():
    from robstride_dynamics.table import (  # type: ignore
        MODEL_MIT_KD_TABLE,
        MODEL_MIT_KP_TABLE,
        MODEL_MIT_POSITION_TABLE,
        MODEL_MIT_TORQUE_TABLE,
        MODEL_MIT_VELOCITY_TABLE,
    )

    model = "rs-04"
    return {
        "comm_type_mit_control": 1,
        "comm_type_feedback": 2,
        "comm_type_enable": 3,
        "comm_type_stop": 4,
        "comm_type_set_zero": 6,
        "comm_type_active_feedback": 24,
        "master_id": 0xFD,
        "p_min": -float(MODEL_MIT_POSITION_TABLE[model]),
        "p_max": float(MODEL_MIT_POSITION_TABLE[model]),
        "v_min": -float(MODEL_MIT_VELOCITY_TABLE[model]),
        "v_max": float(MODEL_MIT_VELOCITY_TABLE[model]),
        "kp_min": 0.0,
        "kp_max": float(MODEL_MIT_KP_TABLE[model]),
        "kd_min": 0.0,
        "kd_max": float(MODEL_MIT_KD_TABLE[model]),
        "tau_min": -float(MODEL_MIT_TORQUE_TABLE[model]),
        "tau_max": float(MODEL_MIT_TORQUE_TABLE[model]),
        "use_float_to_uint": False,
    }


class RobStrideMotorInterface(MotorInterface):
    """One RS04 over the repository's validated official SocketCAN path."""

    def __init__(self, interface, motor_id, bitrate=1_000_000, timeout_s=0.003):
        self.interface = str(interface)
        self.motor_id = int(motor_id)
        self.bitrate = int(bitrate)
        self.timeout_s = float(timeout_s)
        self.proto = official_rs04_protocol()
        self.bus = None
        self._enabled = False
        self._last_state = None

    @property
    def connected(self):
        return self.bus is not None

    @property
    def enabled(self):
        return self._enabled

    def connect(self):
        if self.connected:
            return
        self.bus = SocketCan(
            channel=self.interface,
            bitrate=self.bitrate,
            timeout=self.timeout_s,
            tx_retry_count=1,
        ).open()
        self.bus.configure_feedback_filters((2, 24))

    def disconnect(self):
        if self.bus is None:
            return
        try:
            self.disable()
        finally:
            self.bus.close()
            self.bus = None

    def _management(self, key, payload=None):
        if self.bus is None:
            raise RuntimeError("SocketCAN is not connected")
        can_id = motor_management_can_id(self.motor_id, self.proto, key)
        self.bus.send_raw(can_id, bytes(8) if payload is None else bytes(payload))

    def enable(self):
        self._management("comm_type_enable")
        self._enabled = True

    def disable(self):
        if self.bus is not None:
            try:
                self._management("comm_type_stop")
            finally:
                self._enabled = False

    def send_command(self, command):
        if self.bus is None:
            raise RuntimeError("SocketCAN is not connected")
        if not self._enabled:
            raise RuntimeError("motor is not enabled")
        p = self.proto
        checks = (
            ("q_des", command.q_des, p["p_min"], p["p_max"]),
            ("qd_des", command.qd_des, p["v_min"], p["v_max"]),
            ("kp", command.kp, p["kp_min"], p["kp_max"]),
            ("kd", command.kd, p["kd_min"], p["kd_max"]),
            ("tau_ff", command.tau_ff, p["tau_min"], p["tau_max"]),
        )
        for name, value, lower, upper in checks:
            if not lower <= float(value) <= upper:
                raise ValueError(f"{name}={value} outside RS04 protocol [{lower}, {upper}]")
        self.bus.send_raw(
            mit_can_id(self.motor_id, p, command.tau_ff),
            pack_mit_command(command.q_des, command.qd_des, command.kp, command.kd, p),
        )

    def read_state(self, timeout_s=0.0):
        if self.bus is None:
            return None
        frames = self.bus.read_available_frames(
            timeout=max(0.0, float(timeout_s)),
            max_frames=64,
            expected_motor_ids={self.motor_id},
            feedback_comm_types={2, 24},
        )
        latest = None
        for frame in frames:
            decoded = decode_mit_feedback_frame(frame.can_id, frame.data, self.proto)
            if decoded is not None and int(decoded["motor_id"]) == self.motor_id:
                latest = MotorState(
                    timestamp_monotonic=float(frame.timestamp),
                    position=float(decoded["position"]),
                    velocity=float(decoded["velocity"]),
                    torque_measured=float(decoded["torque"]),
                    current=None,
                    temperature=float(decoded["temperature_c"]),
                    voltage=None,
                    fault_bits=int(decoded["fault_bits"]),
                    mode_status=int(decoded["mode_status"]),
                    feedback_source="rs04_mit_feedback",
                )
        if latest is not None:
            self._last_state = latest
        return latest

    def passive_poll(self):
        """Request feedback while stopped; never enables impedance control."""
        self._management("comm_type_stop")
        return self.read_state(timeout_s=self.timeout_s)
