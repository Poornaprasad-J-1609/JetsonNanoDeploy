#!/usr/bin/env python3
"""Static qualification gates for policy-controlled motor enable."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import yaml

from policy_qualification import check_golden_vectors
from policy_runner import PolicyRunner


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DeploymentReadiness:
    policy_checks: tuple[ReadinessCheck, ...]
    hardware_checks: tuple[ReadinessCheck, ...]

    @property
    def policy_ready(self) -> bool:
        return all(check.passed for check in self.policy_checks)

    @property
    def hardware_ready(self) -> bool:
        return self.policy_ready and all(
            check.passed for check in self.hardware_checks
        )

    def failed(self) -> tuple[ReadinessCheck, ...]:
        return tuple(
            check
            for check in (*self.policy_checks, *self.hardware_checks)
            if not check.passed
        )

    def lines(self) -> list[str]:
        lines = [
            "MODEL_12357 DEPLOYMENT READINESS",
            f"Policy semantics: {'PASS' if self.policy_ready else 'BLOCKED'}",
            f"Hardware enable: {'PASS' if self.hardware_ready else 'BLOCKED'}",
        ]
        for group_name, checks in (
            ("Policy contract", self.policy_checks),
            ("Hardware contract", self.hardware_checks),
        ):
            lines.append(group_name + ":")
            for check in checks:
                status = "PASS" if check.passed else "FAIL"
                lines.append(f"  [{status}] {check.name}: {check.detail}")
        return lines


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_check(mapping: dict, key: str, label: str) -> ReadinessCheck:
    passed = bool(mapping.get(key, False))
    return ReadinessCheck(
        label,
        passed,
        "verified" if passed else f"{key} is not verified",
    )


def _all_profiles_have_limits(four_bar: dict) -> bool:
    profiles = (four_bar.get("four_bar_transmission", {}) or {}).get(
        "profiles", {}
    ) or {}
    if not profiles:
        return False
    return all(
        float(profile.get("min_abs_jacobian", 0.0)) >= 0.05
        and float(profile.get("endpoint_tolerance_rad", 1.0)) <= 0.01
        for profile in profiles.values()
    )


def evaluate_deployment_readiness(
    root: Path,
    policy_path: Path | None = None,
) -> DeploymentReadiness:
    root = Path(root).resolve()
    contract = (
        _load_yaml(root / "config" / "policy_contract.yaml").get(
            "policy_contract", {}
        )
        or {}
    )
    artifact = contract.get("artifact", {}) or {}
    semantic = contract.get("semantic_contract", {}) or {}
    observation = semantic.get("observation", {}) or {}
    joint_order = semantic.get("joint_order", {}) or {}
    action = semantic.get("action", {}) or {}
    hardware = contract.get("hardware_contract", {}) or {}

    policy_path = Path(policy_path or root / "policy" / "policy.pt")
    expected_hash = str(artifact.get("sha256", "")).lower()
    actual_hash = _sha256(policy_path) if policy_path.is_file() else ""
    training_source = semantic.get("training_source_path")
    training_source_path = (
        root / str(training_source) if training_source else None
    )
    golden_vectors = semantic.get("golden_vectors_path")
    golden_path = root / str(golden_vectors) if golden_vectors else None
    golden_passed = False
    golden_detail = (
        str(golden_path) if golden_path else "golden_vectors_path is missing"
    )
    if golden_path and golden_path.is_file() and policy_path.is_file():
        try:
            runner = PolicyRunner(policy_path=policy_path)
            result = check_golden_vectors(
                runner,
                golden_path,
                tolerance=1.0e-5,
            )
            golden_passed = bool(result["passed"])
            golden_detail = (
                f"cases={result.get('case_count', 0)} "
                f"max_abs_error={result['maximum_absolute_error']:.9g}"
            )
            if result["errors"]:
                golden_detail += " errors=" + "; ".join(result["errors"])
        except Exception as exc:
            golden_detail = f"validation failed: {exc}"

    policy_checks = (
        ReadinessCheck(
            "policy artifact SHA256",
            bool(expected_hash and actual_hash == expected_hash),
            f"expected={expected_hash or 'missing'} actual={actual_hash or 'missing'}",
        ),
        ReadinessCheck(
            "exact Isaac training source",
            bool(training_source_path and training_source_path.exists()),
            (
                str(training_source_path)
                if training_source_path
                else "training_source_path is missing"
            ),
        ),
        _bool_check(semantic, "verified", "semantic contract"),
        _bool_check(
            observation,
            "clock_formula_verified",
            "marching clock formula/frequency/reset",
        ),
        _bool_check(
            observation,
            "clock_provider_implemented",
            "marching clock runtime provider",
        ),
        _bool_check(
            observation,
            "previous_action_convention_verified",
            "previous-action convention",
        ),
        _bool_check(
            observation,
            "stationary_command_enforced",
            "stationary Phase-2 command override",
        ),
        _bool_check(
            joint_order,
            "verified_from_training",
            "policy joint order",
        ),
        _bool_check(
            joint_order,
            "position_velocity_signs_verified_from_training",
            "policy joint observation signs",
        ),
        _bool_check(
            action,
            "clip_verified_from_training",
            "actor clipping convention",
        ),
        _bool_check(
            action,
            "scale_verified_from_training",
            "per-index action scale",
        ),
        _bool_check(
            action,
            "per_index_action_pipeline_implemented",
            "per-index clip/scale target pipeline",
        ),
        _bool_check(
            action,
            "q_default_verified_from_training",
            "training default joint pose",
        ),
        ReadinessCheck(
            "independent Isaac golden vectors",
            golden_passed,
            golden_detail,
        ),
    )

    four_bar_cfg = _load_yaml(root / "config" / "four_bar_transmission.yaml")
    four_bar = four_bar_cfg.get("four_bar_transmission", {}) or {}
    four_bar_required = bool(hardware.get("four_bar_required", False))
    four_bar_enabled = bool(four_bar.get("enabled", False))
    four_bar_limits_ok = _all_profiles_have_limits(four_bar_cfg)

    hardware_checks = (
        _bool_check(hardware, "verified", "complete hardware contract"),
        _bool_check(
            hardware,
            "policy_to_motor_mapping_verified",
            "policy-to-motor routing",
        ),
        _bool_check(
            hardware,
            "encoder_zero_and_sign_verified",
            "encoder zero/sign calibration",
        ),
        _bool_check(
            hardware,
            "joint_limits_verified",
            "physical joint limits",
        ),
        ReadinessCheck(
            "required four-bar path enabled",
            (not four_bar_required) or four_bar_enabled,
            f"required={four_bar_required} enabled={four_bar_enabled}",
        ),
        ReadinessCheck(
            "four-bar endpoint/Jacobian guards",
            (not four_bar_required) or four_bar_limits_ok,
            "requires min_abs_jacobian>=0.05 and endpoint_tolerance<=0.01",
        ),
        _bool_check(
            hardware,
            "four_bar_calibration_verified",
            "four-bar calibration",
        ),
        _bool_check(hardware, "imu_frame_verified", "IMU/base frame"),
        _bool_check(
            hardware,
            "imu_rate_and_timestamp_verified",
            "IMU rate/timestamps",
        ),
        _bool_check(
            hardware,
            "effective_joint_gains_verified",
            "effective joint-space gains",
        ),
        _bool_check(
            hardware,
            "torque_current_conversion_verified",
            "torque/current conversion",
        ),
        _bool_check(
            hardware,
            "battery_voltage_monitor_verified",
            "battery undervoltage monitor",
        ),
        _bool_check(
            hardware,
            "thermal_limits_verified",
            "motor thermal limits",
        ),
        _bool_check(
            hardware,
            "low_level_timing_verified",
            "50 Hz/200 Hz timing",
        ),
        _bool_check(
            hardware,
            "safety_state_machine_verified",
            "explicit hardware safety state machine",
        ),
        _bool_check(
            hardware,
            "current_measurement_and_logging_verified",
            "motor current measurement/logging",
        ),
        _bool_check(
            hardware,
            "battery_measurement_and_logging_verified",
            "battery voltage measurement/logging",
        ),
        _bool_check(
            hardware,
            "sustained_saturation_watchdog_verified",
            "sustained torque/current saturation watchdog",
        ),
        _bool_check(
            hardware,
            "unexpected_motor_id_watchdog_verified",
            "unexpected motor-ID watchdog",
        ),
        ReadinessCheck(
            "hardware-enable approval",
            bool(contract.get("hardware_enable_permitted", False)),
            (
                "explicitly permitted"
                if contract.get("hardware_enable_permitted", False)
                else "hardware_enable_permitted is false"
            ),
        ),
    )
    return DeploymentReadiness(policy_checks, hardware_checks)


def failure_lines(checks: Iterable[ReadinessCheck]) -> list[str]:
    return [f"{check.name}: {check.detail}" for check in checks if not check.passed]
