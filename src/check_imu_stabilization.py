#!/usr/bin/env python3
"""Hardware-free end-to-end check for stand IMU posture stabilization."""

import math
import numpy as np

from can_topology import resolve_joint_can_bus
from imu_interface import projected_gravity_absolute_xsens
from main_controller import (
    apply_imu_posture_stabilization,
    imu_posture_correction,
    load_motion_assist_config,
    load_motor_ids,
    projected_gravity_to_roll_pitch,
    stand_policy_imu_correction,
)
from motor_command_layer import MotorCommandLayer
from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor


def gravity_for_tilt(roll_deg=0.0, pitch_deg=0.0):
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    if abs(roll) > 0.0 and abs(pitch) > 0.0:
        raise ValueError("dry checker uses one tilt axis at a time")
    if abs(roll) > 0.0:
        return np.array([0.0, -math.sin(roll), -math.cos(roll)], dtype=np.float32)
    return np.array([math.sin(pitch), 0.0, -math.cos(pitch)], dtype=np.float32)


def settle_through_safety(safety, requested, start, cycles=100):
    target = np.asarray(start, dtype=np.float32).copy()
    for _ in range(int(cycles)):
        target = safety.safety_filter(requested, target)
    return target


def quaternion_for_tilt(roll_deg=0.0, pitch_deg=0.0):
    roll_half = 0.5 * math.radians(float(roll_deg))
    pitch_half = 0.5 * math.radians(float(pitch_deg))
    if abs(roll_half) > 0.0 and abs(pitch_half) > 0.0:
        raise ValueError("dry checker uses one tilt axis at a time")
    if abs(roll_half) > 0.0:
        return np.array([math.cos(roll_half), math.sin(roll_half), 0.0, 0.0])
    return np.array([math.cos(pitch_half), 0.0, math.sin(pitch_half), 0.0])


def main():
    runner = PolicyRunner()
    cfg = load_motion_assist_config()
    cfg.setdefault("imu_posture", {})["enabled"] = True
    safety = SafetyMonitor(runner.policy_order)
    layer = MotorCommandLayer(
        policy_order=runner.policy_order,
        motor_ids=load_motor_ids(),
        active_joints=runner.policy_order,
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 2),
    )

    cases = [
        ("upright", 0.0, 0.0),
        ("roll+5", 5.0, 0.0),
        ("roll-5", -5.0, 0.0),
        ("pitch+5", 0.0, 5.0),
        ("pitch-5", 0.0, -5.0),
        ("roll+20", 20.0, 0.0),
        ("pitch+20", 0.0, 20.0),
    ]
    corrections = {}

    print("GRALLATOR STAND IMU STABILIZATION DRY CHECK")
    print("No serial, CAN, IMU, or motor device is opened.\n")

    for label, roll_deg, pitch_deg in cases:
        expected_gravity = gravity_for_tilt(roll_deg=roll_deg, pitch_deg=pitch_deg)
        quat = quaternion_for_tilt(roll_deg=roll_deg, pitch_deg=pitch_deg)
        xsens_gravity = projected_gravity_absolute_xsens(quat)["projected_gravity"]
        if not np.allclose(xsens_gravity, expected_gravity, atol=1e-6):
            raise AssertionError(f"{label}: Xsens quaternion gravity conversion mismatch")

    for label, roll_deg, pitch_deg in cases:
        gravity = gravity_for_tilt(roll_deg=roll_deg, pitch_deg=pitch_deg)
        measured_roll, measured_pitch = projected_gravity_to_roll_pitch(gravity)
        correction = imu_posture_correction(gravity, runner.policy_order, cfg)
        requested = apply_imu_posture_stabilization(
            runner.q_stand,
            gravity,
            runner.policy_order,
            cfg,
        )
        settled = settle_through_safety(safety, requested, runner.q_stand)
        commands = layer.build_mit_commands(settled, phase="startup")
        corrections[label] = correction

        if not np.all(np.isfinite(settled)):
            raise AssertionError(f"{label}: non-finite stabilization target")
        if np.any(settled < safety.q_min - 1e-6) or np.any(settled > safety.q_max + 1e-6):
            raise AssertionError(f"{label}: target escaped hard joint limits")
        for command in commands:
            expected = command["offset"] + command["direction"] * command["q_des"]
            if abs(float(command["p_base"]) - float(expected)) > 1e-7:
                raise AssertionError(f"{label}: motor direction conversion mismatch")

        moving = [
            f"{name}={settled[index]:+.4f}"
            for index, name in enumerate(runner.policy_order)
            if abs(float(settled[index] - runner.q_stand[index])) > 1e-5
        ]
        print(
            f"{label:9s} tilt_rp=[{math.degrees(measured_roll):+5.1f},"
            f"{math.degrees(measured_pitch):+5.1f}]deg "
            f"corr_max={np.max(np.abs(correction)):.4f} "
            f"moving={len(moving):02d}"
        )
        if moving:
            print("  " + " ".join(moving))

    if np.max(np.abs(corrections["upright"])) > 1e-7:
        raise AssertionError("upright IMU must produce zero posture correction")
    for positive, negative in (("roll+5", "roll-5"), ("pitch+5", "pitch-5")):
        if not np.allclose(corrections[positive], -corrections[negative], atol=1e-7):
            raise AssertionError(f"{positive}/{negative}: corrections are not symmetric")
        if np.max(np.abs(corrections[positive])) < 0.02:
            raise AssertionError(f"{positive}: correction is too small to observe")

    print("\nIMU STABILIZATION DRY CHECK OK")
    print("Upright is neutral; roll/pitch signs are symmetric; targets obey limits and motor directions.")

    print("\nDIFFERENTIAL RL STAND STABILIZATION")
    learned = {}
    zero3 = np.zeros(3, dtype=np.float32)
    zero12 = np.zeros(len(runner.policy_order), dtype=np.float32)
    for label, roll_deg, pitch_deg in cases[:5]:
        gravity = gravity_for_tilt(roll_deg=roll_deg, pitch_deg=pitch_deg)
        correction, _, _, delta_action = stand_policy_imu_correction(
            runner=runner,
            base_ang_vel_b=zero3,
            projected_gravity_b=gravity,
            q_current=runner.q_stand,
            qd_current=zero12,
            cfg=cfg,
        )
        requested = runner.q_stand + correction
        settled = settle_through_safety(safety, requested, runner.q_stand)
        commands = layer.build_mit_commands(settled, phase="startup")
        learned[label] = correction
        if not np.all(np.isfinite(correction)):
            raise AssertionError(f"{label}: differential RL correction is non-finite")
        if np.any(settled < safety.q_min - 1e-6) or np.any(settled > safety.q_max + 1e-6):
            raise AssertionError(f"{label}: differential RL target escaped limits")
        for command in commands:
            expected = command["offset"] + command["direction"] * command["q_des"]
            if abs(float(command["p_base"]) - float(expected)) > 1e-7:
                raise AssertionError(f"{label}: differential RL motor conversion mismatch")
        print(
            f"{label:9s} delta_action_max={np.max(np.abs(delta_action)):.4f} "
            f"joint_corr_max={np.max(np.abs(correction)):.4f}"
        )

    if np.max(np.abs(learned["upright"])) > 1e-7:
        raise AssertionError("upright differential RL correction must be exactly zero")
    for positive, negative in (("roll+5", "roll-5"), ("pitch+5", "pitch-5")):
        norm_product = float(np.linalg.norm(learned[positive]) * np.linalg.norm(learned[negative]))
        opposition = float(np.dot(learned[positive], -learned[negative]) / norm_product)
        if opposition < 0.90:
            raise AssertionError(f"{positive}/{negative}: learned responses do not oppose")
        if np.max(np.abs(learned[positive])) < 0.02:
            raise AssertionError(f"{positive}: learned correction is too small")
        print(f"  {positive}/{negative} opposition={opposition:.3f}")

    print("DIFFERENTIAL RL STAND STABILIZATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
