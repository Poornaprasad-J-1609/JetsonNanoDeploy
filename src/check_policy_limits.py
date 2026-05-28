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
from motor_command_layer import MotorCommandLayer
from joystick_interface import (
    clip_command,
    load_command_limits,
    load_joystick_defaults,
    load_speed_scale_defaults,
)


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6


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
    q_safe = safety.safety_filter(q_raw, runner.q_default)
    return command_clipped, action, q_raw, q_hard, q_safe


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

        if abs(float(cmd["p_des"]) - (q_des + float(cmd["offset"]))) > 2e-5:
            violations.append(
                f"{label}: {joint_name} p_des does not match q_des + offset"
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
        active_joints=motor_cfg.get("active_joints", []),
        joint_can_bus=motor_cfg.get("joint_can_bus", {}),
    )
    limits = load_yaml(ROOT / "config" / "joint_limits.yaml")["joint_limits"]

    print("Policy:", runner.policy_path)
    print("Observation/action dim:", runner.observation_dim, runner.action_dim)
    print("Action scale:", runner.action_scale)
    print("Control dt:", runner.control_dt)

    failures = []
    failures += check_pose("default_pose", runner.q_default, runner, limits)
    failures += check_pose("stand_pose", runner.q_stand, runner, limits)
    failures += check_pose("crouch_pose", runner.q_crouch, runner, limits)
    print_control_limits()

    print("\nRate limits:")
    for i, joint_name in enumerate(runner.policy_order):
        print(
            f"  {joint_name:16s} {safety.dq_max[i]:.3f} rad/step "
            f"= {safety.dq_max[i] / runner.control_dt:.2f} rad/s"
        )

    sample_commands = [
        ("forward", [0.30, 0.0, 0.0]),
        ("fast_fwd", [0.45, 0.0, 0.0]),
        ("backward", [-0.30, 0.0, 0.0]),
        ("left", [0.0, 0.25, 0.0]),
        ("right", [0.0, -0.25, 0.0]),
        ("yaw_left", [0.0, 0.0, 0.45]),
        ("yaw_right", [0.0, 0.0, -0.45]),
        ("diagonal", [0.30, 0.20, 0.30]),
        ("too_fast", [3.00, -3.00, 3.00]),
    ]

    print("\nSample policy targets and final MIT command targets from default pose:")
    position_clip_count = 0
    runtime_violations = []
    for label, command in sample_commands:
        command = np.asarray(command, dtype=np.float32)
        command_clipped, action, q_raw, q_hard, q_safe = sample_policy(
            command,
            runner,
            safety,
            args.base_lin_vel_source,
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

    if failures:
        print("\nFAIL: pose outside joint limits:", sorted(set(failures)))
        return 1
    if runtime_violations:
        print("\nFAIL: runtime limit violations:")
        for violation in runtime_violations:
            print(" ", violation)
        return 1

    print("\nOK: poses are inside limits.")
    print("OK: velocity commands are clipped to control_limits.yaml before policy observation.")
    print("OK: safety_filter enforces configured joint position/rate limits when enabled.")
    print("OK: MotorCommandLayer hard-clips every final MIT q_des to joint_limits.yaml.")
    print("OK: MIT kp/kd/v/tau parameters obey control_limits.yaml when mit_parameters.enabled is true.")
    if position_clip_count > 0:
        print("Note: policy_raw position clips were observed, but final motor commands stayed inside limits.")
    print("Note: rate clips are target slew limits from dq_max_per_step, not hard joint angle limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
