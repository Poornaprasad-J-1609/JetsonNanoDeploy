from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MotorCommand:
    q_des: float = 0.0
    qd_des: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    tau_ff: float = 0.0


@dataclass(frozen=True)
class MotorState:
    timestamp_monotonic: float
    position: float
    velocity: float
    torque_measured: float | None = None
    current: float | None = None
    temperature: float | None = None
    voltage: float | None = None
    fault_bits: int = 0
    mode_status: int = 0
    feedback_source: str = "unknown"


class MotorInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def enable(self) -> None: ...

    @abstractmethod
    def disable(self) -> None: ...

    @abstractmethod
    def send_command(self, command: MotorCommand) -> None: ...

    @abstractmethod
    def read_state(self, timeout_s: float = 0.0) -> MotorState | None: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    @property
    @abstractmethod
    def enabled(self) -> bool: ...
