from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class MotorConfig:
    model: str = "rs-04"
    id: int = 1
    interface: str = "slcan0"
    bitrate: int = 1_000_000
    feedback_timeout_s: float = 0.10
    torque_feedback_source: str = "rs04_mit_feedback"
    current_source: str = "unavailable"
    voltage_source: str = "unavailable"
    torque_constant_nm_per_a: float | None = None


@dataclass(frozen=True)
class ControlConfig:
    frequency_hz: float = 200.0
    gui_frequency_hz: float = 25.0
    initial_kp: float = 40.0
    initial_kd: float = 2.0
    kp_min: float = 0.0
    kp_max: float = 500.0
    kd_min: float = 0.0
    kd_max: float = 10.0
    tau_ff_min_nm: float = -20.0
    tau_ff_max_nm: float = 20.0
    manual_step_rad: float = 0.017453292519943295
    manual_speed_rad_s: float = 0.25


@dataclass(frozen=True)
class SafetyConfig:
    min_position_rad: float = -2.5
    max_position_rad: float = 2.5
    max_velocity_rad_s: float = 8.0
    max_torque_nm: float = 40.0
    max_current_a: float = 30.0
    max_temperature_c: float = 70.0
    communication_timeout_s: float = 0.10
    feedback_timeout_s: float = 0.10
    max_consecutive_late_cycles: int = 10


@dataclass(frozen=True)
class LoggingConfig:
    directory: str = "logs/rs04_bench"
    queue_size: int = 20_000
    flush_interval_s: float = 0.5


@dataclass(frozen=True)
class AnalysisConfig:
    filter_window: int = 21
    filter_polynomial_order: int = 3
    velocity_deadband_rad_s: float = 0.03
    minimum_identification_samples: int = 200
    minimum_excitation_velocity_rad_s: float = 0.15
    settling_band_fraction: float = 0.02


@dataclass(frozen=True)
class PendulumConfig:
    attached_mass_kg: float = 0.0
    mass_com_radius_m: float = 0.0
    lever_mass_kg: float = 0.0
    lever_com_radius_m: float = 0.0
    known_lever_inertia_kg_m2: float = 0.0
    gravity_m_s2: float = 9.80665
    angle_zero_convention: str = "downward_vertical"


@dataclass(frozen=True)
class MockConfig:
    inertia_kg_m2: float = 0.020
    viscous_damping_nm_s_rad: float = 0.12
    coulomb_friction_nm: float = 0.15
    command_delay_s: float = 0.010
    torque_limit_nm: float = 40.0
    initial_temperature_c: float = 28.0


@dataclass(frozen=True)
class BenchConfig:
    motor: MotorConfig = field(default_factory=MotorConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    pendulum: PendulumConfig = field(default_factory=PendulumConfig)
    mock: MockConfig = field(default_factory=MockConfig)
    source_path: str = ""

    @property
    def period_s(self) -> float:
        return 1.0 / self.control.frequency_hz


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> BenchConfig:
    source = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if source != DEFAULT_CONFIG_PATH.resolve():
        with source.open("r", encoding="utf-8") as stream:
            data = _merge(data, yaml.safe_load(stream) or {})
    if overrides:
        data = _merge(data, overrides)
    config = BenchConfig(
        motor=MotorConfig(**data["motor"]),
        control=ControlConfig(**data["control"]),
        safety=SafetyConfig(**data["safety"]),
        logging=LoggingConfig(**data["logging"]),
        analysis=AnalysisConfig(**data["analysis"]),
        pendulum=PendulumConfig(**data["pendulum"]),
        mock=MockConfig(**data["mock"]),
        source_path=str(source),
    )
    validate_config(config)
    return config


def validate_config(config: BenchConfig) -> None:
    if config.motor.model.lower() not in {"rs-04", "rs04"}:
        raise ValueError("This bench currently supports RobStride RS04 only")
    if not 1 <= config.motor.id <= 0xFF:
        raise ValueError("motor.id must be within [1, 255]")
    if config.motor.torque_constant_nm_per_a is not None and config.motor.torque_constant_nm_per_a <= 0:
        raise ValueError("motor.torque_constant_nm_per_a must be positive when configured")
    if config.control.frequency_hz != 200.0:
        raise ValueError("control.frequency_hz must be exactly 200 Hz")
    if config.control.kp_min < 0 or config.control.kp_max > 5000:
        raise ValueError("configured Kp range is outside the official RS04 protocol")
    if config.control.kd_min < 0 or config.control.kd_max > 100:
        raise ValueError("configured Kd range is outside the official RS04 protocol")
    if config.control.kp_min > config.control.kp_max or config.control.kd_min > config.control.kd_max:
        raise ValueError("gain minimum exceeds maximum")
    if not 0 < config.control.manual_speed_rad_s <= config.safety.max_velocity_rad_s:
        raise ValueError("control.manual_speed_rad_s must be positive and within the velocity limit")
    if config.safety.min_position_rad >= config.safety.max_position_rad:
        raise ValueError("position safety limits are invalid")
    if config.analysis.filter_window < 5 or config.analysis.filter_window % 2 == 0:
        raise ValueError("analysis.filter_window must be an odd integer >= 5")
    if config.pendulum.angle_zero_convention not in {"downward_vertical", "horizontal"}:
        raise ValueError("pendulum.angle_zero_convention must be downward_vertical or horizontal")
