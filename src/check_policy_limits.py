#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor
from motor_command_layer import MotorCommandLayer, motor_position_to_joint_angle
from imu_interface import projected_gravity_absolute_xsens
from main_controller import (
    filtered_policy_action,
    is_recoverable_feedback_issue,
    rate_limit_policy_command,
    run_dry_policy_contract_check,
    smoothstep,
)
from state_estimator import FakeStateEstimator
from joystick_interface import (
    clip_command,
    keyboard_motion_command,
    load_command_convention,
    load_command_limits,
    load_joystick_defaults,
    load_speed_scale_defaults,
)


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6
SIMULATION_NATIVE_JOINT_ORDER = [
    "BL_hip_joint",
    "BR_hip_joint",
    "FL_hip_joint",
    "FR_hip_joint",
    "BL_thigh_joint",
    "BR_thigh_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "BL_calf_joint",
    "BR_calf_joint",
    "FL_calf_joint",
    "FR_calf_joint",
]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def check_pose(name, q, runner, limits):
    failures = []
    print(f"\n{name}:")
    for i, joint_name in enumerate(runner.policy_order):
        lo = float(limits[joint_name]["min"])
        hi = float(limits[joint_name]["max"])
        value = float(q[i])
        ok = lo <= value <= hi
        status = "OK" if ok else "OUT"
        print(
            f"  {joint_name:16s} q={value:+.3f} "
            f"limit=[{lo:+.3f},{hi:+.3f}] {status}"
        )
        if not ok:
            failures.append(joint_name)
    return failures


def array_limits_by_joint(runner, limits):
    q_min = np.array([float(limits[name]["min"]) for name in runner.policy_order], dtype=np.float32)
    q_max = np.array([float(limits[name]["max"]) for name in runner.policy_order], dtype=np.float32)
    dq_max = np.array(
        [float(limits[name]["dq_max_per_step"]) for name in runner.policy_order],
        dtype=np.float32,
    )
    return q_min, q_max, dq_max


def check_observation_layout(runner):
    failures = []
    expected_lengths = {
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command": 3,
        "joint_pos_relative": len(runner.policy_order),
        "joint_vel": len(runner.policy_order),
        "previous_action": len(runner.policy_order),
    }
    used = []

    print("\nObservation layout:")
    for field_name, expected_len in expected_lengths.items():
        if field_name not in runner.observation_layout:
            failures.append(f"missing observation field: {field_name}")
            print(f"  {field_name:20s} MISSING")
            continue
        try:
            indices = runner.observation_layout[field_name]
            indices = list(range(indices[0], indices[1] + 1)) if len(indices) == 2 else list(indices)
        except Exception as exc:
            failures.append(f"{field_name}: invalid layout: {exc}")
            print(f"  {field_name:20s} INVALID {exc}")
            continue
        if len(indices) != expected_len:
            failures.append(
                f"{field_name}: layout has {len(indices)} slots, expected {expected_len}"
            )
        if any(index < 0 or index >= runner.observation_dim for index in indices):
            failures.append(f"{field_name}: index outside observation dim {runner.observation_dim}")
        used.extend(indices)
        print(f"  {field_name:20s} indices={indices[0]}..{indices[-1]} count={len(indices)}")

    duplicates = sorted({index for index in used if used.count(index) > 1})
    if duplicates:
        failures.append(f"duplicate observation indices: {duplicates}")
    missing = [index for index in range(runner.observation_dim) if index not in used]
    if missing:
        failures.append(f"unused observation indices: {missing}")
    return failures


def check_simulation_joint_order(runner):
    print("\nPolicy joint order from simulation telemetry:")
    for index, joint_name in enumerate(runner.policy_order):
        print(f"  {index:02d}: {joint_name}")
    if list(runner.policy_order) != SIMULATION_NATIVE_JOINT_ORDER:
        return [
            "policy joint order differs from the native IsaacLab telemetry order"
        ]
    return []


def check_imu_upright_frame():
    upright = projected_gravity_absolute_xsens(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )["projected_gravity"]
    print("\nIMU frame sanity:")
    print(f"  identity quaternion projected_gravity={upright}")
    if not np.allclose(upright, np.array([0.0, 0.0, -1.0], dtype=np.float32), atol=1e-5):
        return ["Xsens upright projected_gravity is not [0, 0, -1]"]
    return []


def check_policy_joint_radian_source(runner):
    failures = []
    estimator = FakeStateEstimator(runner.q_default)
    if getattr(estimator, "joint_state_units", None) != "joint_radians":
        failures.append("state estimator does not label policy state as joint radians")
    q_joint, qd_joint = estimator.read_policy_joint_state()
    if not np.array_equal(q_joint, runner.q_default):
        failures.append("policy joint-state API returned unexpected joint positions")
    if not np.array_equal(qd_joint, np.zeros(12, dtype=np.float32)):
        failures.append("policy joint-state API returned unexpected joint velocities")

    expected_joint_angle = 0.37
    offset = -0.24
    conversion_errors = []
    for direction in (1.0, -1.0):
        raw_motor_angle = offset + direction * (expected_joint_angle + 2.0 * np.pi)
        converted = motor_position_to_joint_angle(
            raw_motor_angle,
            offset=offset,
            direction=direction,
        )
        conversion_errors.append(abs(converted - expected_joint_angle))
    max_error = max(conversion_errors)
    if max_error > 1e-6:
        failures.append(
            f"raw motor to joint-radian conversion error is {max_error:.9f} rad"
        )

    print("\nPolicy joint-state source:")
    print("  source=converted joint radians")
    print(f"  +/- direction and 2*pi conversion max_error={max_error:.9f} rad")
    print("  raw motor position is retained only in feedback['position_raw']")
    return failures


def check_feedback_hold_classification():
    recoverable = (
        "ABNORMAL ENCODER ANGLE: missing MIT encoder feedback before motion",
        "ABNORMAL ENCODER ANGLE: stale MIT encoder feedback before motion",
    )
    hard_stop = (
        "MOTOR FAULT: nonzero MIT feedback fault bits",
        "ABNORMAL ENCODER ANGLE: FR_calf_joint=+8.0 rad",
    )
    failures = []
    if not all(is_recoverable_feedback_issue(reason) for reason in recoverable):
        failures.append("missing/stale feedback is not classified as FEEDBACK HOLD")
    if any(is_recoverable_feedback_issue(reason) for reason in hard_stop):
        failures.append("motor/encoder safety fault was incorrectly classified as recoverable")
    print("\nFeedback failure behavior:")
    print("  missing/stale feedback -> HOLD")
    print("  motor fault/impossible angle -> EMERGENCY STOP")
    return failures


def check_keyboard_mapping():
    convention = load_command_convention()
    left_y = 1.2 if convention["vy_left_positive"] else -1.2
    expected = {
        "W": (["w"], [2.2, 0.0, 0.0]),
        "S": (["s"], [-2.2, 0.0, 0.0]),
        "A": (["a"], [0.0, left_y, 0.0]),
        "D": (["d"], [0.0, -left_y, 0.0]),
        "Q": (["q"], [0.0, 0.0, 0.7]),
        "E": (["e"], [0.0, 0.0, -0.7]),
        "W+D": (["w", "d"], [2.2, -left_y, 0.0]),
        "W+A": (["w", "a"], [2.2, left_y, 0.0]),
        "S+A": (["s", "a"], [-2.2, left_y, 0.0]),
        "S+D": (["s", "d"], [-2.2, -left_y, 0.0]),
        "W+Q": (["w", "q"], [2.2, 0.0, 0.7]),
        "W+S": (["w", "s"], [0.0, 0.0, 0.0]),
        "A+D": (["a", "d"], [0.0, 0.0, 0.0]),
        "Q+E": (["q", "e"], [0.0, 0.0, 0.0]),
    }
    failures = []
    print("\nKeyboard body-frame mapping:")
    for label, (keys, expected_command) in expected.items():
        command = keyboard_motion_command(
            keys,
            2.2,
            1.2,
            0.7,
            vy_left_positive=convention["vy_left_positive"],
        )
        expected_array = np.asarray(expected_command, dtype=np.float32)
        ok = bool(np.allclose(command, expected_array, atol=1e-7))
        print(f"  {label:3s} -> {command.tolist()} {'OK' if ok else 'WRONG'}")
        if not ok:
            failures.append(
                f"keyboard {label}: got {command.tolist()}, expected {expected_command}"
            )
    return failures


def within_limits(q, q_min, q_max, eps=EPS):
    q = np.asarray(q, dtype=np.float32)
    return np.logical_and(q >= q_min - eps, q <= q_max + eps)


def print_control_limits():
    command_limits = load_command_limits()
    speed_scale = load_speed_scale_defaults()
    joystick_defaults = load_joystick_defaults()

    print("\nVelocity command limits from config/control_limits.yaml:")
    print(
        f"  vx=[{command_limits[0]:+.3f},{command_limits[1]:+.3f}] "
        f"vy=[{command_limits[2]:+.3f},{command_limits[3]:+.3f}] "
        f"yaw=[{command_limits[4]:+.3f},{command_limits[5]:+.3f}]"
    )
    print(
        "Joystick speed scale:",
        f"initial={speed_scale['initial']:.2f}",
        f"min={speed_scale['min']:.2f}",
        f"max={speed_scale['max']:.2f}",
        f"step={speed_scale['step']:.2f}",
    )
    print(
        "Joystick full-stick before speed scale:",
        f"vx={float(joystick_defaults['speed_limits']['max_vx']):.3f}",
        f"vy={float(joystick_defaults['speed_limits']['max_vy']):.3f}",
        f"yaw={float(joystick_defaults['speed_limits']['max_yaw']):.3f}",
    )
    return command_limits


def sample_policy(command, runner, safety, base_lin_vel_source):
    command = np.asarray(command, dtype=np.float32)
    command_clipped = clip_command(command, load_command_limits())
    if base_lin_vel_source == "command":
        base_lin_vel = np.array([command_clipped[0], command_clipped[1], 0.0], dtype=np.float32)
    elif base_lin_vel_source == "zero":
        base_lin_vel = np.zeros(3, dtype=np.float32)
    else:
        raise ValueError("check script supports base_lin_vel_source command or zero")

    obs = runner.build_observation(
        base_lin_vel_b=base_lin_vel,
        base_ang_vel_b=np.zeros(3, dtype=np.float32),
        projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        command=command_clipped,
        q_current=runner.q_default,
        qd_current=np.zeros(len(runner.policy_order), dtype=np.float32),
        previous_action=np.zeros(len(runner.policy_order), dtype=np.float32),
    )
    action = runner.infer_action(obs)
    q_raw = runner.action_to_q_target(action)
    q_hard = np.clip(q_raw, safety.q_min, safety.q_max)
    q_safe = safety.safety_filter(
        q_raw,
        runner.q_default,
        rate_profile="policy",
    )
    return command_clipped, obs, action, q_raw, q_hard, q_safe


def check_motor_commands(label, q_target, runner, motor_layer, limits, phase="policy", verbose=False):
    q_min, q_max, _ = array_limits_by_joint(runner, limits)
    limit_by_joint = {
        name: (float(q_min[i]), float(q_max[i]))
        for i, name in enumerate(runner.policy_order)
    }
    commands = motor_layer.build_mit_commands(q_target, phase=phase)
    violations = []

    for cmd in commands:
        joint_name = cmd["joint_name"]
        q_lo, q_hi = limit_by_joint[joint_name]
        q_des = float(cmd["q_des"])
        if q_des < q_lo - EPS or q_des > q_hi + EPS:
            violations.append(
                f"{label}: {joint_name} q_des={q_des:+.4f} outside [{q_lo:+.4f},{q_hi:+.4f}]"
            )

        expected_p = float(cmd["offset"]) + float(cmd.get("direction", 1.0)) * q_des
        if abs(float(cmd["p_des"]) - expected_p) > 2e-5:
            violations.append(
                f"{label}: {joint_name} p_des does not match direction*q_des + offset"
            )

        if motor_layer.mit_parameter_limits_enabled:
            lim = motor_layer.mit_parameter_limits
            parameter_checks = {
                "p_des": (float(cmd["p_des"]), lim["p_min"], lim["p_max"]),
                "v_des": (float(cmd["v_des"]), lim["v_min"], lim["v_max"]),
                "kp": (float(cmd["kp"]), lim["kp_min"], lim["kp_max"]),
                "kd": (float(cmd["kd"]), lim["kd_min"], lim["kd_max"]),
                "tau_ff": (float(cmd["tau_ff"]), lim["tau_ff_min"], lim["tau_ff_max"]),
            }
            for field, (value, lo, hi) in parameter_checks.items():
                if value < lo - EPS or value > hi + EPS:
                    violations.append(
                        f"{label}: {joint_name} {field}={value:+.4f} outside [{lo:+.4f},{hi:+.4f}]"
                    )

    if verbose:
        print(f"\nMIT command check: {label}")
        for cmd in commands:
            print(
                f"  {cmd['joint_name']:16s} id=0x{cmd['motor_id']:02X} "
                f"q_req={cmd['q_requested']:+.4f} q_des={cmd['q_des']:+.4f} "
                f"kp={cmd['kp']:.2f} kd={cmd['kd']:.2f} tau={cmd['tau_ff']:.2f}"
            )

    return violations


def check_extreme_target_enforcement(runner, safety, motor_layer, limits):
    q_min, q_max, dq_max = array_limits_by_joint(runner, limits)
    signs = np.where(np.arange(len(runner.policy_order)) % 2 == 0, 1.0, -1.0).astype(np.float32)
    q_extreme = runner.q_default + signs * 100.0
    q_safe = safety.safety_filter(q_extreme, runner.q_default)

    violations = []
    final_ok = within_limits(q_safe, q_min, q_max)
    if safety.joint_position_enabled and not bool(np.all(final_ok)):
        for i, ok in enumerate(final_ok):
            if not ok:
                violations.append(
                    f"safety_filter extreme target: {runner.policy_order[i]} "
                    f"q={q_safe[i]:+.4f} outside [{q_min[i]:+.4f},{q_max[i]:+.4f}]"
                )

    dq = np.abs(q_safe - runner.q_default)
    rate_ok = dq <= dq_max + EPS
    if safety.joint_rate_enabled and not bool(np.all(rate_ok)):
        for i, ok in enumerate(rate_ok):
            if not ok:
                violations.append(
                    f"safety_filter extreme target: {runner.policy_order[i]} "
                    f"dq={dq[i]:.4f} > dq_max={dq_max[i]:.4f}"
                )

    violations.extend(
        check_motor_commands(
            label="direct extreme target into MotorCommandLayer",
            q_target=q_extreme,
            runner=runner,
            motor_layer=motor_layer,
            limits=limits,
            phase="policy",
        )
    )
    violations.extend(
        check_motor_commands(
            label="safety-filtered extreme target into MotorCommandLayer",
            q_target=q_safe,
            runner=runner,
            motor_layer=motor_layer,
            limits=limits,
            phase="policy",
        )
    )

    max_final_step = float(np.max(dq))
    print("\nExtreme target enforcement:")
    print(
        f"  safety_filter max target step={max_final_step:.4f} rad "
        f"(joint_rate={'on' if safety.joint_rate_enabled else 'off'})"
    )
    print("  MotorCommandLayer direct extreme target clamp: checked")
    return violations


def check_closed_loop_gait_passthrough(runner, safety):
    """Verify healthy mid-speed rollouts do not touch deployment guards."""
    cfg = load_yaml(ROOT / "config" / "control_limits.yaml")
    deployment = cfg.get("policy_deployment", {})
    speed = cfg.get("joystick_speed_scale", {})
    joystick = load_joystick_defaults()["speed_limits"]
    convention = load_command_convention()
    left_sign = 1.0 if convention["vy_left_positive"] else -1.0
    scale = float(speed.get("initial", 0.10))
    cases = {
        "W": [float(joystick["max_vx"]) * scale, 0.0, 0.0],
        "S": [-float(joystick["max_vx"]) * scale, 0.0, 0.0],
        "A": [0.0, left_sign * float(joystick["max_vy"]) * scale, 0.0],
        "D": [0.0, -left_sign * float(joystick["max_vy"]) * scale, 0.0],
        "Q": [0.0, 0.0, float(joystick["max_yaw"]) * scale],
        "E": [0.0, 0.0, -float(joystick["max_yaw"]) * scale],
        "W+A": [float(joystick["max_vx"]) * scale, left_sign * float(joystick["max_vy"]) * scale, 0.0],
        "W+D": [float(joystick["max_vx"]) * scale, -left_sign * float(joystick["max_vy"]) * scale, 0.0],
        "S+A": [-float(joystick["max_vx"]) * scale, left_sign * float(joystick["max_vy"]) * scale, 0.0],
        "S+D": [-float(joystick["max_vx"]) * scale, -left_sign * float(joystick["max_vy"]) * scale, 0.0],
    }

    clip_abs = float(deployment.get("action_clip_abs", 0.0))
    smoothing = float(deployment.get("action_smoothing", 0.0))
    transition_steps = max(
        0,
        int(np.ceil(float(deployment.get("transition_seconds", 0.0)) / runner.control_dt)),
    )
    failures = []

    print("\nClosed-loop mid-speed gait guard check:")
    print("  case  peak_action action_clips hard_clips rate_clips target_abs_max")
    for label, target_command in cases.items():
        estimator = FakeStateEstimator(runner.q_default)
        previous_action = np.zeros(len(runner.policy_order), dtype=np.float32)
        previous_target = runner.q_default.copy()
        filtered_command = np.zeros(3, dtype=np.float32)
        peak_action = 0.0
        action_clips = 0
        hard_clips = 0
        rate_clips = 0
        target_abs_max = 0.0

        for step in range(300):
            q_current, qd_current, _, _, _ = estimator.read()
            filtered_command = rate_limit_policy_command(
                target=np.asarray(target_command, dtype=np.float32),
                previous=filtered_command,
                dt=runner.control_dt,
                vx_per_s=float(deployment.get("command_slew_vx_per_s", 0.0)),
                vy_per_s=float(deployment.get("command_slew_vy_per_s", 0.0)),
                yaw_per_s=float(deployment.get("command_slew_yaw_per_s", 0.0)),
            )
            obs = runner.build_observation(
                base_lin_vel_b=np.zeros(3, dtype=np.float32),
                base_ang_vel_b=np.zeros(3, dtype=np.float32),
                projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                command=filtered_command,
                q_current=q_current,
                qd_current=qd_current,
                previous_action=previous_action,
            )
            raw_action = runner.infer_action(obs)
            action = filtered_policy_action(
                raw_action,
                previous_action,
                clip_abs=clip_abs,
                smoothing=smoothing,
            )
            peak_action = max(peak_action, float(np.max(np.abs(raw_action))))
            action_clips += int(not np.allclose(action, raw_action, atol=EPS))

            q_policy = runner.action_to_q_target(action)
            if transition_steps > 0 and step < transition_steps:
                alpha = smoothstep(float(step + 1) / float(transition_steps))
                q_policy = (alpha * q_policy).astype(np.float32)
            q_hard = np.clip(q_policy, safety.q_min, safety.q_max)
            hard_clips += int(not np.allclose(q_hard, q_policy, atol=EPS))
            q_safe = safety.safety_filter(
                q_policy,
                previous_target,
                rate_profile="policy",
            )
            rate_clips += int(not np.allclose(q_safe, q_hard, atol=EPS))
            target_abs_max = max(target_abs_max, float(np.max(np.abs(q_safe))))

            estimator.dry_update_as_if_robot_followed(q_safe, runner.control_dt)
            previous_action = action.copy()
            previous_target = q_safe.copy()

        print(
            f"  {label:4s} {peak_action:11.3f} {action_clips:12d} "
            f"{hard_clips:10d} {rate_clips:10d} {target_abs_max:14.3f}"
        )
        if action_clips:
            failures.append(f"{label}: healthy rollout touched policy action cap")
        if hard_clips:
            failures.append(f"{label}: healthy rollout touched a hard joint limit")
        if rate_clips:
            failures.append(f"{label}: healthy rollout touched policy target-rate limits")

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-lin-vel-source",
        choices=["command", "zero"],
        default="zero",
    )
    args = parser.parse_args()

    runner = PolicyRunner()
    safety = SafetyMonitor(runner.policy_order)
    motor_cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    motor_layer = MotorCommandLayer(
        policy_order=runner.policy_order,
        motor_ids=motor_cfg["motor_ids"],
        active_joints=runner.policy_order,
        joint_can_bus=motor_cfg.get("joint_can_bus", {}),
    )
    limits = load_yaml(ROOT / "config" / "joint_limits.yaml")["joint_limits"]

    print("Policy:", runner.policy_path)
    print("Policy SHA256:", runner.policy_sha256)
    print("Observation/action dim:", runner.observation_dim, runner.action_dim)
    print("Action scale:", runner.action_scale)
    print("Control dt:", runner.control_dt)
    print("Force zero base linear velocity:", runner.force_zero_base_linear_velocity)

    failures = []
    if not runner.policy_path.exists():
        failures.append(f"policy path does not exist: {runner.policy_path}")
    if runner.policy_format not in ("torchscript", "checkpoint_actor"):
        failures.append(f"unsupported policy format: {runner.policy_format}")
    if runner.observation_dim != 48:
        failures.append(f"policy observation dim is {runner.observation_dim}, expected 48")
    if runner.action_dim != 12:
        failures.append(f"policy action dim is {runner.action_dim}, expected 12")
    if len(runner.policy_order) != 12 or len(set(runner.policy_order)) != 12:
        failures.append("policy order is not exactly 12 unique joints")
    if runner.q_default.shape != (12,) or not np.all(np.isfinite(runner.q_default)):
        failures.append("q_default is not a finite 12-value vector")
    if not np.isfinite(runner.action_scale) or runner.action_scale <= 0.0:
        failures.append("policy action scale is not finite and positive")
    if not runner.force_zero_base_linear_velocity:
        failures.append("policy contract does not force base linear velocity to zero")

    command_sign_failures = check_keyboard_mapping()
    failures += command_sign_failures
    motion_cfg = load_yaml(ROOT / "config" / "motion_assist.yaml")
    motion_assists_default_disabled = not bool(
        motion_cfg.get("imu_posture", {}).get("enabled", False)
        or motion_cfg.get("gait_assist", {}).get("enabled", False)
    )
    if not motion_assists_default_disabled:
        failures.append("motion assists are enabled by default")

    can_writes = []
    motor_layer.send_signal_commands = lambda *args, **kwargs: can_writes.append("mit")
    motor_layer.send_harmless_frames = lambda *args, **kwargs: can_writes.append("signal")
    try:
        dry_result = run_dry_policy_contract_check(
            runner=runner,
            safety=safety,
            motor_layer=motor_layer,
            emit=False,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        failures.append(f"dry-run policy contract failed: {exc}")
        dry_result = None
    dry_run_no_can = bool(
        dry_result is not None
        and dry_result["dry_run_no_can"]
        and not can_writes
    )
    if not dry_run_no_can:
        failures.append("dry-run attempted a CAN write")
    base_lin_vel_zero = bool(
        dry_result is not None
        and np.array_equal(
            dry_result["observation"][0:3],
            np.zeros(3, dtype=np.float32),
        )
    )
    if not base_lin_vel_zero:
        failures.append("dry-run observation base linear velocity is not exactly zero")
    joint_radians_to_policy = bool(
        dry_result is not None
        and dry_result.get("joint_radians_to_policy", False)
    )
    if not joint_radians_to_policy:
        failures.append("dry-run policy state was not converted joint radians")
    command_names = (
        [command["joint_name"] for command in dry_result["commands"]]
        if dry_result is not None
        else []
    )
    if command_names != list(runner.policy_order) or len(command_names) != 12:
        failures.append("motor command layer did not receive 12 commands in policy order")

    failures += check_observation_layout(runner)
    failures += check_simulation_joint_order(runner)
    failures += check_imu_upright_frame()
    failures += check_policy_joint_radian_source(runner)
    feedback_hold_failures = check_feedback_hold_classification()
    failures += feedback_hold_failures
    failures += check_pose("default_pose", runner.q_default, runner, limits)
    failures += check_pose("stand_pose", runner.q_stand, runner, limits)
    failures += check_pose("crouch_pose", runner.q_crouch, runner, limits)
    failures += check_pose("sit_pose_when_stand_zero", runner.q_sit_when_stand_zero, runner, limits)
    failures += check_pose("stand_pose_when_sit_zero", runner.q_stand_when_sit_zero, runner, limits)
    print_control_limits()

    print("\nRate limits:")
    print(
        "  pose profile:",
        "enabled" if safety.joint_rate_enabled else "disabled",
        "| policy profile:",
        "enabled" if safety.policy_joint_rate_enabled else "disabled",
    )
    for i, joint_name in enumerate(runner.policy_order):
        print(
            f"  {joint_name:16s} pose={safety.dq_max[i]:.3f} rad/step "
            f"({safety.dq_max[i] / runner.control_dt:.2f} rad/s) "
            f"policy={safety.dq_max_policy[i]:.3f} rad/step "
            f"({safety.dq_max_policy[i] / runner.control_dt:.2f} rad/s)"
        )

    sample_commands = [
        ("forward", [0.30, 0.0, 0.0]),
        ("fast_fwd", [0.45, 0.0, 0.0]),
        ("backward", [-0.30, 0.0, 0.0]),
        ("positive_y", [0.0, 0.25, 0.0]),
        ("negative_y", [0.0, -0.25, 0.0]),
        ("positive_yaw", [0.0, 0.0, 0.45]),
        ("negative_yaw", [0.0, 0.0, -0.45]),
        ("W+A", [0.30, 0.20, 0.0]),
        ("W+D", [0.30, -0.20, 0.0]),
        ("S+A", [-0.30, 0.20, 0.0]),
        ("S+D", [-0.30, -0.20, 0.0]),
        ("too_fast", [3.00, -3.00, 3.00]),
    ]

    print("\nSample policy targets and final MIT command targets from default pose:")
    position_clip_count = 0
    runtime_violations = []
    for label, command in sample_commands:
        command = np.asarray(command, dtype=np.float32)
        command_clipped, obs, action, q_raw, q_hard, q_safe = sample_policy(
            command,
            runner,
            safety,
            args.base_lin_vel_source,
        )
        if runner.force_zero_base_linear_velocity and not np.allclose(
            obs[0:3],
            np.zeros(3, dtype=np.float32),
        ):
            runtime_violations.append(
                f"{label}: base linear velocity observation is not forced to zero"
            )
        pos_clipped = np.abs(q_hard - q_raw) > 1e-6
        rate_clipped = np.abs(q_safe - q_hard) > 1e-6
        command_was_clipped = np.any(np.abs(command_clipped - command) > 1e-6)
        position_clip_count += int(pos_clipped.sum())
        runtime_violations.extend(
            check_motor_commands(
                label=label,
                q_target=q_safe,
                runner=runner,
                motor_layer=motor_layer,
                limits=limits,
                phase="policy",
            )
        )
        print(
            f"  {label:9s} cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}] "
            f"sent=[{command_clipped[0]:+.2f},{command_clipped[1]:+.2f},{command_clipped[2]:+.2f}] "
            f"cmd_clip={'yes' if command_was_clipped else 'no'} "
            f"action_abs_max={np.max(np.abs(action)):.3f} "
            f"position_clips={int(pos_clipped.sum())} "
            f"rate_clips={int(rate_clipped.sum())}"
        )
        for i, joint_name in enumerate(runner.policy_order):
            if pos_clipped[i]:
                print(
                    f"    POSITION CLIP {joint_name:16s} "
                    f"policy_raw={q_raw[i]:+.3f} hard_clip={q_hard[i]:+.3f}"
                )

    runtime_violations.extend(check_extreme_target_enforcement(runner, safety, motor_layer, limits))
    runtime_violations.extend(check_closed_loop_gait_passthrough(runner, safety))

    if failures:
        print("\nFAIL: pose outside joint limits:", sorted(set(failures)))
        return 1
    if runtime_violations:
        print("\nFAIL: runtime limit violations:")
        for violation in runtime_violations:
            print(" ", violation)
        return 1

    print("\nPolicy contract acceptance:")
    print("PASS observation_dim=48")
    print("PASS action_dim=12")
    print(f"PASS base_lin_vel_zero={base_lin_vel_zero}")
    print("PASS policy_order_12_unique=True")
    print("PASS q_default_valid=True")
    print("PASS action_scale_valid=True")
    print(f"PASS command_signs_consistent={not command_sign_failures}")
    print(
        "PASS motion_assists_default_disabled="
        f"{motion_assists_default_disabled}"
    )
    print(f"PASS dry_run_no_can={dry_run_no_can}")
    print(f"PASS joint_radians_to_policy={joint_radians_to_policy}")
    print(f"PASS stale_feedback_enters_hold={not feedback_hold_failures}")

    print("\nOK: poses are inside limits.")
    print("OK: velocity commands are clipped to control_limits.yaml before policy observation.")
    print("OK: safety_filter enforces hard positions and each enabled rate profile.")
    print("OK: MotorCommandLayer hard-clips every final MIT q_des to joint_limits.yaml.")
    print("OK: MIT kp/kd/v/tau parameters obey control_limits.yaml when mit_parameters.enabled is true.")
    if position_clip_count > 0:
        print("Note: policy_raw position clips were observed, but final motor commands stayed inside limits.")
    print("Note: rate clips are target slew limits from dq_max_per_step, not hard joint angle limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
