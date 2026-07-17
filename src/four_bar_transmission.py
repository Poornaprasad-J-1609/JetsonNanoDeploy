#!/usr/bin/env python3
"""Measured nonlinear motor-to-virtual-knee transmission.

The policy remains in the Isaac/URDF virtual calf coordinate. The physical
motor is converted only at the command and encoder-feedback boundaries.

Coordinate convention:
    theta_m = motor_direction * (raw_motor_angle - joint_offset)
    q_k     = virtual Isaac/URDF calf angle
    J       = dq_k / dtheta_m

Mappings:
    qdot_k = J * theta_dot_m
    tau_k  = efficiency * tau_m / J
    tau_m  = J * tau_k / efficiency
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - deployment dependency message
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


class TransmissionConfigurationError(ValueError):
    """The calibration file is incomplete or mechanically inconsistent."""


class TransmissionRangeError(RuntimeError):
    """A command or measurement is outside the measured calibration range."""


class TransmissionSingularityError(RuntimeError):
    """The linkage Jacobian is too close to zero for safe operation."""


def _finite_1d(name: str, values: Iterable[float], minimum_size: int = 3) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size < minimum_size:
        raise TransmissionConfigurationError(
            f"{name} needs at least {minimum_size} measured samples; got {arr.size}"
        )
    if not np.all(np.isfinite(arr)):
        raise TransmissionConfigurationError(f"{name} contains NaN or Inf")
    return arr


def _monotonic_direction(values: np.ndarray) -> int:
    delta = np.diff(values)
    if np.all(delta > 0.0):
        return 1
    if np.all(delta < 0.0):
        return -1
    return 0


@dataclass(frozen=True)
class LookupProfile:
    name: str
    motor_angle_rad: np.ndarray
    knee_angle_rad: np.ndarray
    efficiency: float = 1.0
    min_abs_jacobian: float = 0.05
    motor_torque_limit_nm: float = 120.0
    endpoint_tolerance_rad: float = 0.01
    clamp_outside_calibration: bool = False
    compensate_efficiency_in_commands: bool = True

    @classmethod
    def from_config(cls, name: str, cfg: Mapping[str, object]) -> "LookupProfile":
        theta = _finite_1d(
            f"profiles.{name}.motor_angle_rad", cfg.get("motor_angle_rad", [])
        )
        q = _finite_1d(
            f"profiles.{name}.knee_angle_rad", cfg.get("knee_angle_rad", [])
        )
        if theta.shape != q.shape:
            raise TransmissionConfigurationError(
                f"profiles.{name}: motor and knee sample counts differ "
                f"({theta.size} != {q.size})"
            )

        # np.interp requires an increasing x-axis. Sort by motor angle while
        # preserving the paired measured knee values.
        order = np.argsort(theta)
        theta = theta[order]
        q = q[order]
        if _monotonic_direction(theta) != 1:
            raise TransmissionConfigurationError(
                f"profiles.{name}.motor_angle_rad must be strictly monotonic"
            )
        if _monotonic_direction(q) == 0:
            raise TransmissionConfigurationError(
                f"profiles.{name}.knee_angle_rad must be strictly monotonic; "
                "the table cannot cross a toggle or switch assembly branch"
            )

        slope = np.diff(q) / np.diff(theta)
        if not np.all(np.isfinite(slope)):
            raise TransmissionConfigurationError(
                f"profiles.{name}: invalid finite-difference Jacobian"
            )

        efficiency = float(cfg.get("efficiency", 1.0))
        min_abs_jacobian = float(cfg.get("min_abs_jacobian", 0.05))
        motor_torque_limit_nm = float(cfg.get("motor_torque_limit_nm", 120.0))
        endpoint_tolerance_rad = float(cfg.get("endpoint_tolerance_rad", 0.01))
        if not np.isfinite(efficiency) or not (0.0 < efficiency <= 1.0):
            raise TransmissionConfigurationError(
                f"profiles.{name}.efficiency must be in (0, 1]"
            )
        if not np.isfinite(min_abs_jacobian) or min_abs_jacobian <= 0.0:
            raise TransmissionConfigurationError(
                f"profiles.{name}.min_abs_jacobian must be finite and > 0"
            )
        if not np.isfinite(motor_torque_limit_nm) or motor_torque_limit_nm <= 0.0:
            raise TransmissionConfigurationError(
                f"profiles.{name}.motor_torque_limit_nm must be finite and > 0"
            )
        if not np.isfinite(endpoint_tolerance_rad) or endpoint_tolerance_rad < 0.0:
            raise TransmissionConfigurationError(
                f"profiles.{name}.endpoint_tolerance_rad must be finite and >= 0"
            )
        measured_min_j = float(np.min(np.abs(slope)))
        if measured_min_j < min_abs_jacobian:
            raise TransmissionConfigurationError(
                f"profiles.{name}: measured |dq/dtheta| reaches "
                f"{measured_min_j:.6f}, below min_abs_jacobian="
                f"{min_abs_jacobian}. Trim the range away from the toggle or "
                "lower the threshold only after mechanical review."
            )

        return cls(
            name=str(name),
            motor_angle_rad=theta,
            knee_angle_rad=q,
            efficiency=efficiency,
            min_abs_jacobian=min_abs_jacobian,
            motor_torque_limit_nm=motor_torque_limit_nm,
            endpoint_tolerance_rad=endpoint_tolerance_rad,
            clamp_outside_calibration=bool(
                cfg.get("clamp_outside_calibration", False)
            ),
            compensate_efficiency_in_commands=bool(
                cfg.get("compensate_efficiency_in_commands", True)
            ),
        )

    @property
    def motor_min(self) -> float:
        return float(self.motor_angle_rad[0])

    @property
    def motor_max(self) -> float:
        return float(self.motor_angle_rad[-1])

    @property
    def knee_min(self) -> float:
        return float(np.min(self.knee_angle_rad))

    @property
    def knee_max(self) -> float:
        return float(np.max(self.knee_angle_rad))

    def _bounded(self, value: float, low: float, high: float, quantity: str) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise TransmissionRangeError(f"{self.name}: {quantity} is NaN or Inf")
        if low <= value <= high:
            return value
        tolerance = float(self.endpoint_tolerance_rad)
        if low - tolerance <= value < low:
            return low
        if high < value <= high + tolerance:
            return high
        if self.clamp_outside_calibration:
            return float(np.clip(value, low, high))
        raise TransmissionRangeError(
            f"{self.name}: {quantity}={value:.6f} is outside measured range "
            f"[{low:.6f}, {high:.6f}] with endpoint_tolerance_rad="
            f"{tolerance:.6f}"
        )

    def knee_from_motor(self, motor_angle: float) -> float:
        theta = self._bounded(
            motor_angle, self.motor_min, self.motor_max, "motor angle"
        )
        return float(np.interp(theta, self.motor_angle_rad, self.knee_angle_rad))

    def motor_from_knee(self, knee_angle: float) -> float:
        q = self._bounded(knee_angle, self.knee_min, self.knee_max, "knee angle")
        if self.knee_angle_rad[0] < self.knee_angle_rad[-1]:
            q_axis = self.knee_angle_rad
            theta_axis = self.motor_angle_rad
        else:
            q_axis = self.knee_angle_rad[::-1]
            theta_axis = self.motor_angle_rad[::-1]
        return float(np.interp(q, q_axis, theta_axis))

    def jacobian_from_motor(self, motor_angle: float) -> float:
        theta = self._bounded(
            motor_angle, self.motor_min, self.motor_max, "motor angle"
        )
        index = int(np.searchsorted(self.motor_angle_rad, theta, side="right") - 1)
        index = int(np.clip(index, 0, self.motor_angle_rad.size - 2))
        dtheta = self.motor_angle_rad[index + 1] - self.motor_angle_rad[index]
        dq = self.knee_angle_rad[index + 1] - self.knee_angle_rad[index]
        jacobian = float(dq / dtheta)
        if abs(jacobian) < self.min_abs_jacobian:
            raise TransmissionSingularityError(
                f"{self.name}: |J|={abs(jacobian):.6f} is below "
                f"min_abs_jacobian={self.min_abs_jacobian}"
            )
        return jacobian


@dataclass(frozen=True)
class JointTransmission:
    joint_name: str
    profile: LookupProfile
    virtual_sign: float = 1.0

    def __post_init__(self) -> None:
        if float(self.virtual_sign) not in (-1.0, 1.0):
            raise TransmissionConfigurationError(
                f"{self.joint_name}: virtual_sign must be exactly +1 or -1"
            )

    def virtual_from_motor(self, motor_angle: float) -> float:
        return float(self.virtual_sign) * self.profile.knee_from_motor(motor_angle)

    def motor_from_virtual(self, virtual_angle: float) -> float:
        return self.profile.motor_from_knee(
            float(self.virtual_sign) * float(virtual_angle)
        )

    def jacobian_from_motor(self, motor_angle: float) -> float:
        return float(self.virtual_sign) * self.profile.jacobian_from_motor(motor_angle)

    def jacobian_from_virtual(self, virtual_angle: float) -> float:
        return self.jacobian_from_motor(self.motor_from_virtual(virtual_angle))


class FourBarTransmissionSet:
    """Per-joint nonlinear transmissions with identity fallback."""

    def __init__(
        self,
        enabled: bool,
        joints: Optional[Mapping[str, JointTransmission]] = None,
        require_feedback_for_commands: bool = True,
        clamp_policy_to_hard_limits: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self._joints: Dict[str, JointTransmission] = dict(joints or {})
        self.require_feedback_for_commands = bool(require_feedback_for_commands)
        self.clamp_policy_to_hard_limits = bool(clamp_policy_to_hard_limits)

    @classmethod
    def from_yaml(
        cls, path: Path, policy_order: Optional[Iterable[str]] = None
    ) -> "FourBarTransmissionSet":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Four-bar config not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        cfg = raw.get("four_bar_transmission", raw) or {}
        enabled = bool(cfg.get("enabled", False))
        require_feedback = bool(cfg.get("require_feedback_for_commands", True))
        clamp_policy = bool(cfg.get("clamp_policy_to_hard_limits", True))
        if not enabled:
            return cls(
                enabled=False,
                joints={},
                require_feedback_for_commands=require_feedback,
                clamp_policy_to_hard_limits=clamp_policy,
            )

        profiles_cfg = cfg.get("profiles", {}) or {}
        joints_cfg = cfg.get("joints", {}) or {}
        required_profiles = {
            str((item or {}).get("profile", "")).strip()
            for item in joints_cfg.values()
            if bool((item or {}).get("enabled", True))
        }
        required_profiles.discard("")
        profiles = {}
        for profile_name in sorted(required_profiles):
            if profile_name not in profiles_cfg:
                raise TransmissionConfigurationError(
                    f"profile '{profile_name}' is referenced by an enabled joint "
                    "but is not defined"
                )
            profiles[profile_name] = LookupProfile.from_config(
                profile_name,
                profiles_cfg.get(profile_name) or {},
            )
        joints: Dict[str, JointTransmission] = {}
        for joint_name, item in joints_cfg.items():
            item = item or {}
            if not bool(item.get("enabled", True)):
                continue
            profile_name = str(item.get("profile", "")).strip()
            if profile_name not in profiles:
                raise TransmissionConfigurationError(
                    f"{joint_name}: profile '{profile_name}' is not defined"
                )
            joints[str(joint_name)] = JointTransmission(
                joint_name=str(joint_name),
                profile=profiles[profile_name],
                virtual_sign=float(item.get("virtual_sign", 1.0)),
            )

        if not joints:
            raise TransmissionConfigurationError(
                "four_bar_transmission.enabled is true but no enabled joints exist"
            )
        if policy_order is not None:
            policy_names = set(policy_order)
            unknown = sorted(set(joints) - policy_names)
            if unknown:
                raise TransmissionConfigurationError(
                    "Four-bar config contains joints not in policy_order: "
                    + ", ".join(unknown)
                )
        return cls(
            enabled=True,
            joints=joints,
            require_feedback_for_commands=require_feedback,
            clamp_policy_to_hard_limits=clamp_policy,
        )

    def is_enabled(self, joint_name: str) -> bool:
        return self.enabled and joint_name in self._joints

    def _joint(self, joint_name: str) -> JointTransmission:
        if not self.is_enabled(joint_name):
            raise KeyError(f"No enabled four-bar transmission for {joint_name}")
        return self._joints[joint_name]

    def virtual_from_motor(self, joint_name: str, motor_angle: float) -> float:
        if not self.is_enabled(joint_name):
            return float(motor_angle)
        return self._joint(joint_name).virtual_from_motor(motor_angle)

    def motor_from_virtual(self, joint_name: str, virtual_angle: float) -> float:
        if not self.is_enabled(joint_name):
            return float(virtual_angle)
        return self._joint(joint_name).motor_from_virtual(virtual_angle)

    def jacobian_from_motor(self, joint_name: str, motor_angle: float) -> float:
        if not self.is_enabled(joint_name):
            return 1.0
        return self._joint(joint_name).jacobian_from_motor(motor_angle)

    def jacobian_from_virtual(self, joint_name: str, virtual_angle: float) -> float:
        if not self.is_enabled(joint_name):
            return 1.0
        return self._joint(joint_name).jacobian_from_virtual(virtual_angle)

    def efficiency(self, joint_name: str) -> float:
        if not self.is_enabled(joint_name):
            return 1.0
        return float(self._joint(joint_name).profile.efficiency)

    def motor_torque_limit(self, joint_name: str) -> float:
        if not self.is_enabled(joint_name):
            return float("inf")
        return float(self._joint(joint_name).profile.motor_torque_limit_nm)

    def virtual_velocity_from_motor(
        self, joint_name: str, motor_angle: float, motor_velocity: float
    ) -> float:
        return self.jacobian_from_motor(joint_name, motor_angle) * float(motor_velocity)

    def motor_velocity_from_virtual(
        self, joint_name: str, motor_angle: float, virtual_velocity: float
    ) -> float:
        return float(virtual_velocity) / self.jacobian_from_motor(
            joint_name, motor_angle
        )

    def virtual_torque_from_motor(
        self, joint_name: str, motor_angle: float, motor_torque: float
    ) -> float:
        jacobian = self.jacobian_from_motor(joint_name, motor_angle)
        return self.efficiency(joint_name) * float(motor_torque) / jacobian

    def motor_torque_from_virtual(
        self, joint_name: str, motor_angle: float, virtual_torque: float
    ) -> float:
        jacobian = self.jacobian_from_motor(joint_name, motor_angle)
        joint = self._joint(joint_name) if self.is_enabled(joint_name) else None
        compensate = bool(
            joint is not None and joint.profile.compensate_efficiency_in_commands
        )
        denominator = self.efficiency(joint_name) if compensate else 1.0
        return jacobian * float(virtual_torque) / denominator

    def motor_gains_from_virtual(
        self,
        joint_name: str,
        motor_angle: float,
        kp_virtual: float,
        kd_virtual: float,
    ) -> tuple[float, float]:
        jacobian = self.jacobian_from_motor(joint_name, motor_angle)
        joint = self._joint(joint_name) if self.is_enabled(joint_name) else None
        compensate = bool(
            joint is not None and joint.profile.compensate_efficiency_in_commands
        )
        denominator = self.efficiency(joint_name) if compensate else 1.0
        scale = jacobian * jacobian / denominator
        return scale * float(kp_virtual), scale * float(kd_virtual)

    def max_virtual_torque_from_motor_limit(
        self, joint_name: str, motor_angle: float
    ) -> float:
        if not self.is_enabled(joint_name):
            return float("inf")
        jacobian = self.jacobian_from_motor(joint_name, motor_angle)
        return (
            self.efficiency(joint_name)
            * self.motor_torque_limit(joint_name)
            / abs(jacobian)
        )

    def decode_feedback(
        self,
        joint_name: str,
        position_raw: float,
        velocity_raw: float,
        torque_raw: float,
        offset: float,
        direction: float,
    ) -> dict:
        direction = float(direction)
        motor_position = direction * (float(position_raw) - float(offset))
        motor_velocity = direction * float(velocity_raw)
        motor_torque = direction * float(torque_raw)
        joint_position = self.virtual_from_motor(joint_name, motor_position)
        jacobian = self.jacobian_from_motor(joint_name, motor_position)
        joint_velocity = jacobian * motor_velocity
        joint_torque = self.virtual_torque_from_motor(
            joint_name, motor_position, motor_torque
        )
        return {
            "motor_position": motor_position,
            "motor_velocity": motor_velocity,
            "motor_torque": motor_torque,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "joint_torque": joint_torque,
            "transmission_jacobian": jacobian,
            "transmission_efficiency": self.efficiency(joint_name),
            "transmission_enabled": self.is_enabled(joint_name),
        }
