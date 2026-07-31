#!/usr/bin/env python3
"""Replay IsaacLab policy-input CSV rows and compare deployment outputs."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor
from joystick_interface import clip_command, load_command_limits
from main_controller import clip_policy_hip_actions, filtered_policy_action
from motor_command_layer import MotorCommandLayer, joint_group


def floats(row, prefix, count, width):
    return np.asarray(
        [float(row[f"{prefix}_{index:0{width}d}"]) for index in range(count)],
        dtype=np.float32,
    )


def floats_from_candidates(row, candidates, count):
    for prefix, width in candidates:
        keys = [f"{prefix}_{index:0{width}d}" for index in range(count)]
        if all(key in row and row[key] != "" for key in keys):
            return np.asarray([float(row[key]) for key in keys], dtype=np.float32)
    expected = ", ".join(prefix for prefix, _ in candidates)
    raise KeyError(f"CSV row has none of the expected vector prefixes: {expected}")


def first_float(row, keys):
    for key in keys:
        if key in row and row[key] != "":
            return float(row[key])
    raise KeyError(f"CSV row has none of the expected fields: {', '.join(keys)}")


def command_label(command, threshold=1e-6):
    labels = []
    vx, vy, yaw = [float(value) for value in command]
    if vx > threshold:
        labels.append("forward")
    elif vx < -threshold:
        labels.append("backward")
    if vy > threshold:
        labels.append("left")
    elif vy < -threshold:
        labels.append("right")
    if yaw > threshold:
        labels.append("yaw_left")
    elif yaw < -threshold:
        labels.append("yaw_right")
    return "+".join(labels) if labels else "stop"


def command_segments(rows, actor_commands):
    segments = []
    start = 0
    for index in range(1, len(rows) + 1):
        changed = index == len(rows)
        if not changed:
            changed = not np.allclose(
                actor_commands[index],
                actor_commands[start],
                atol=1e-6,
                rtol=0.0,
            )
        if changed:
            first = rows[start]
            last = rows[index - 1]
            command = actor_commands[start]
            segments.append({
                "id": len(segments),
                "label": command_label(command),
                "step_start": int(first["step"]),
                "step_end": int(last["step"]),
                "time_start": float(first["sim_time"]),
                "time_end": float(last["sim_time"]),
                "vx": float(command[0]),
                "vy": float(command[1]),
                "yaw": float(command[2]),
                "rows": index - start,
            })
            start = index
    return segments


def load_motor_rows(path):
    by_step_joint = {}
    order = {}
    row_count_by_step = {}
    with Path(path).open(newline="") as motor_file:
        for row in csv.DictReader(motor_file):
            step = int(row["step"])
            joint_name = row["joint_name"]
            by_step_joint[(step, joint_name)] = row
            order[int(row["joint_index"])] = joint_name
            row_count_by_step[step] = row_count_by_step.get(step, 0) + 1
    return by_step_joint, [order[index] for index in sorted(order)], row_count_by_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_input_log")
    parser.add_argument("motor_log")
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-policy-mismatch",
        action="store_true",
        help="write comparison output and return success even when logged and deployed actors differ",
    )
    parser.add_argument(
        "--force-zero-base-lin-vel",
        action="store_true",
        help=(
            "set obs[0:3] to zero before replay. Use this only to analyze "
            "simulation logs that were accidentally recorded with nonzero "
            "base linear velocity for the 48-slot zero-base policy contract"
        ),
    )
    parser.add_argument(
        "--policy-sim-match",
        action="store_true",
        help=(
            "compare the raw sim-matching runtime path: no action shaping, "
            "no target rate limiting, and no software PD torque target rewrite"
        ),
    )
    args = parser.parse_args()

    runner = PolicyRunner(policy_path=args.policy_path)
    safety = SafetyMonitor(runner.policy_order)
    with Path(args.policy_input_log).open(newline="") as policy_file:
        rows = list(csv.DictReader(policy_file))
    if not rows:
        raise ValueError("Policy input log has no rows")

    motor_rows, sim_joint_order, motor_row_count_by_step = load_motor_rows(args.motor_log)
    if sim_joint_order != list(runner.policy_order):
        raise ValueError(
            "Joint order mismatch:\n"
            f"  simulation={sim_joint_order}\n"
            f"  deployment={list(runner.policy_order)}"
        )

    observations = np.stack([floats(row, "obs", 48, 3) for row in rows])
    original_base_lin_vel_abs_max = float(np.max(np.abs(observations[:, 0:3])))
    if args.force_zero_base_lin_vel:
        observations[:, 0:3] = 0.0
    sim_actions = np.stack([
        floats_from_candidates(
            row,
            (("action", 2), ("policy_action", 2)),
            12,
        )
        for row in rows
    ])
    logged_commands = np.asarray(
        [[float(row["command_vx"]), float(row["command_vy"]), float(row["command_yaw"])] for row in rows],
        dtype=np.float32,
    )
    # The exact actor input is authoritative. Descriptive command columns can
    # lag keyboard state or contain stale values in simulation loggers.
    commands = observations[:, 9:12].copy()
    command_limits = load_command_limits()
    runtime_commands = np.stack(
        [clip_command(command, command_limits) for command in commands]
    )

    with torch.no_grad():
        replay_actions = runner.policy(torch.from_numpy(observations)).cpu().numpy()
    replay_targets = (
        runner.q_default[None, :]
        + runner.action_scale * replay_actions
    )
    final_targets = np.clip(
        replay_targets,
        safety.policy_q_min[None, :],
        safety.policy_q_max[None, :],
    )

    sim_targets = np.full_like(replay_targets, np.nan)
    sim_positions = np.full_like(replay_targets, np.nan)
    sim_velocities = np.full_like(replay_targets, np.nan)
    sim_motor_actions = np.full_like(replay_actions, np.nan)
    sim_defaults = np.full_like(replay_targets, np.nan)
    computed_torques = np.full_like(replay_targets, np.nan)
    applied_torques = np.full_like(replay_targets, np.nan)
    for row_index, policy_row in enumerate(rows):
        step = int(policy_row["step"])
        for joint_index, joint_name in enumerate(runner.policy_order):
            motor_row = motor_rows.get((step, joint_name))
            if motor_row is None:
                continue
            sim_targets[row_index, joint_index] = first_float(
                motor_row,
                ("joint_pos_target", "q_target_est"),
            )
            sim_positions[row_index, joint_index] = float(motor_row["q"])
            sim_velocities[row_index, joint_index] = float(motor_row["qd"])
            sim_motor_actions[row_index, joint_index] = first_float(
                motor_row,
                ("action", "raw_action_by_joint", "policy_action_same_index"),
            )
            sim_defaults[row_index, joint_index] = float(motor_row["default_q"])
            if "computed_torque_preclip" in motor_row:
                computed_torques[row_index, joint_index] = float(
                    motor_row["computed_torque_preclip"]
                )
            elif "computed_torque" in motor_row:
                computed_torques[row_index, joint_index] = float(
                    motor_row["computed_torque"]
                )
            if "applied_torque_postclip" in motor_row:
                applied_torques[row_index, joint_index] = float(
                    motor_row["applied_torque_postclip"]
                )
            elif "applied_torque" in motor_row:
                applied_torques[row_index, joint_index] = float(
                    motor_row["applied_torque"]
                )

    action_error = np.abs(replay_actions - sim_actions)
    target_error = np.abs(replay_targets - sim_targets)
    final_target_error = np.abs(final_targets - sim_targets)
    sim_position_limit_error = np.maximum(
        np.maximum(safety.q_min[None, :] - sim_positions, 0.0),
        np.maximum(sim_positions - safety.q_max[None, :], 0.0),
    )
    motor_action_error = np.abs(sim_motor_actions - sim_actions)
    episode_reset_rows = np.asarray([
        float(row.get("episode_reset", 0.0) or 0.0) > 0.5
        for row in rows
    ])
    non_reset_rows = ~episode_reset_rows
    command_metadata_error = np.abs(commands - logged_commands)
    previous_action_error = np.abs(observations[1:, 36:48] - sim_actions[:-1])
    reset_rows = (
        (np.max(np.abs(observations[1:, 36:48]), axis=1) <= 1e-7)
        & (np.max(np.abs(sim_actions[:-1]), axis=1) > 1e-5)
    )
    continuity_rows = ~reset_rows
    previous_action_continuity_max = (
        float(np.max(previous_action_error[continuity_rows]))
        if np.any(continuity_rows)
        else 0.0
    )
    runtime_command_clip_error = np.abs(runtime_commands - commands)

    control_cfg = runner.root / "config" / "control_limits.yaml"
    with control_cfg.open("r") as config_file:
        deployment_cfg = yaml.safe_load(config_file)["policy_deployment"]
    action_clip = float(deployment_cfg.get("action_clip_abs", 0.0))
    hip_action_clip = float(deployment_cfg.get("hip_action_clip_abs", 0.0))
    action_smoothing = float(deployment_cfg.get("action_smoothing", 0.0))
    action_delta_limit = float(deployment_cfg.get("action_delta_limit_abs", 0.0))
    if args.policy_sim_match:
        action_clip = 0.0
        hip_action_clip = 0.0
        action_smoothing = 0.0
        action_delta_limit = 0.0

    shaped_actions = np.zeros_like(replay_actions)
    shaped_logged_actions = np.zeros_like(sim_actions)
    previous_shaped = np.zeros(replay_actions.shape[1], dtype=np.float32)
    previous_logged_shaped = np.zeros(sim_actions.shape[1], dtype=np.float32)
    for index in range(len(rows)):
        replay_control_action = clip_policy_hip_actions(
            replay_actions[index],
            runner.policy_order,
            hip_clip_abs=hip_action_clip,
        )
        logged_control_action = clip_policy_hip_actions(
            sim_actions[index],
            runner.policy_order,
            hip_clip_abs=hip_action_clip,
        )
        shaped_actions[index] = filtered_policy_action(
            replay_control_action,
            previous_shaped,
            clip_abs=action_clip,
            smoothing=action_smoothing,
            delta_limit_abs=action_delta_limit,
        )
        shaped_logged_actions[index] = filtered_policy_action(
            logged_control_action,
            previous_logged_shaped,
            clip_abs=action_clip,
            smoothing=action_smoothing,
            delta_limit_abs=action_delta_limit,
        )
        previous_shaped = shaped_actions[index]
        previous_logged_shaped = shaped_logged_actions[index]

    shaped_targets = (
        runner.q_default[None, :]
        + runner.action_scale * shaped_actions
    )
    shaped_logged_targets = (
        runner.q_default[None, :]
        + runner.action_scale * shaped_logged_actions
    )
    shaped_policy_targets = np.clip(
        shaped_logged_targets,
        safety.policy_q_min[None, :],
        safety.policy_q_max[None, :],
    )
    if args.policy_sim_match:
        runtime_targets = shaped_policy_targets.copy()
    else:
        runtime_targets = np.zeros_like(shaped_policy_targets)
        previous_target = runner.q_default.copy()
        for index in range(len(rows)):
            runtime_targets[index] = safety.safety_filter(
                shaped_logged_targets[index],
                previous_target,
                apply_rate_limit=True,
                use_policy_limits=True,
            )
            previous_target = runtime_targets[index]

    action_shaping_error = np.abs(shaped_logged_actions - sim_actions)
    hard_limit_error = np.abs(shaped_policy_targets - shaped_logged_targets)
    rate_limit_error = np.abs(runtime_targets - shaped_policy_targets)
    total_runtime_target_error = np.abs(runtime_targets - sim_targets)

    motor_cfg = yaml.safe_load(
        (runner.root / "config" / "motor_ids.yaml").read_text()
    )
    motor_layer = MotorCommandLayer(
        policy_order=runner.policy_order,
        motor_ids=motor_cfg["motor_ids"],
        active_joints=runner.policy_order,
    )
    directions = np.asarray(
        [motor_layer.joint_directions[name] for name in runner.policy_order],
        dtype=np.float32,
    )
    offsets = np.asarray(
        [motor_layer.joint_offsets[name] for name in runner.policy_order],
        dtype=np.float32,
    )
    motor_targets = offsets[None, :] + directions[None, :] * runtime_targets
    joint_roundtrip = directions[None, :] * (motor_targets - offsets[None, :])
    sign_roundtrip_error = np.abs(joint_roundtrip - runtime_targets)
    motor_torques = directions[None, :] * applied_torques
    torque_roundtrip = directions[None, :] * motor_torques
    torque_sign_roundtrip_error = np.abs(torque_roundtrip - applied_torques)

    gain_fits = []
    for joint_index, joint_name in enumerate(runner.policy_order):
        valid = (
            np.isfinite(sim_targets[:, joint_index])
            & np.isfinite(sim_positions[:, joint_index])
            & np.isfinite(sim_velocities[:, joint_index])
            & np.isfinite(computed_torques[:, joint_index])
        )
        if not np.any(valid):
            gain_fits.append((np.nan, np.nan, np.nan))
            continue
        position_error = (
            sim_targets[valid, joint_index] - sim_positions[valid, joint_index]
        )
        velocity_error = -sim_velocities[valid, joint_index]
        matrix = np.column_stack([
            position_error,
            velocity_error,
            np.ones(np.count_nonzero(valid)),
        ])
        torque = computed_torques[valid, joint_index]
        coefficients = np.linalg.lstsq(matrix, torque, rcond=None)[0]
        estimate = matrix @ coefficients
        denominator = max(float(np.sum((torque - np.mean(torque)) ** 2)), 1e-12)
        r_squared = 1.0 - float(np.sum((torque - estimate) ** 2)) / denominator
        gain_fits.append((
            float(coefficients[0]),
            float(coefficients[1]),
            r_squared,
        ))

    sim_match_targets = np.clip(
        sim_targets,
        safety.policy_q_min[None, :],
        safety.policy_q_max[None, :],
    )
    sim_match_torque_estimate = np.full_like(sim_match_targets, np.nan)
    policy_command_proto = motor_layer.command_proto_for_phase("policy")
    for joint_index, joint_name in enumerate(runner.policy_order):
        group = joint_group(joint_name)
        configured_kp, configured_kd = motor_layer._joint_gains(
            "policy",
            joint_name,
            group,
        )
        effective_kp = motor_layer._effective_unsigned_wire_value(
            configured_kp,
            "kp",
            policy_command_proto,
        )
        effective_kd = motor_layer._effective_unsigned_wire_value(
            configured_kd,
            "kd",
            policy_command_proto,
        )
        valid = (
            np.isfinite(sim_match_targets[:, joint_index])
            & np.isfinite(sim_positions[:, joint_index])
            & np.isfinite(sim_velocities[:, joint_index])
        )
        policy_torque_limit = motor_layer.policy_pd_torque_limit_for_joint(joint_name)
        q_feedback = sim_positions[valid, joint_index]
        qd_feedback = sim_velocities[valid, joint_index]
        velocity_torque = -effective_kd * qd_feedback
        position_torque = effective_kp * (
            sim_match_targets[valid, joint_index] - q_feedback
        )
        if policy_torque_limit > 0.0:
            position_torque = np.clip(
                position_torque,
                -policy_torque_limit - velocity_torque,
                policy_torque_limit - velocity_torque,
            )
        limited_target = q_feedback + position_torque / effective_kp
        sim_match_targets[valid, joint_index] = np.clip(
            limited_target,
            safety.policy_q_min[joint_index],
            safety.policy_q_max[joint_index],
        )
        torque_estimate = (
            effective_kp * (
                sim_match_targets[valid, joint_index] - q_feedback
            )
            + velocity_torque
        )
        if policy_torque_limit > 0.0 and effective_kd > 0.0:
            target_torque = np.clip(
                torque_estimate,
                -policy_torque_limit,
                policy_torque_limit,
            )
            velocity_target = qd_feedback + (
                target_torque
                - effective_kp * (
                    sim_match_targets[valid, joint_index] - q_feedback
                )
            ) / effective_kd
            velocity_target = np.clip(
                velocity_target,
                float(motor_layer.proto["v_min"]),
                float(motor_layer.proto["v_max"]),
            )
            torque_estimate = (
                effective_kp * (
                    sim_match_targets[valid, joint_index] - q_feedback
                )
                + effective_kd * (velocity_target - qd_feedback)
            )
        sim_match_torque_estimate[valid, joint_index] = torque_estimate

    print("Policy:", runner.policy_path)
    print("Policy SHA256:", runner.policy_sha256)
    print("Rows:", len(rows))
    print("Joint order:", ", ".join(sim_joint_order))
    print("Segments:")
    for segment in command_segments(rows, commands):
        duration = segment["rows"] * runner.control_dt
        print(
            f"  {segment['id']:>2} {segment['label']:9s} "
            f"steps={segment['step_start']:04d}..{segment['step_end']:04d} "
            f"duration={duration:.2f}s "
            f"cmd=[{segment['vx']:+.3f},{segment['vy']:+.3f},{segment['yaw']:+.3f}]"
        )

    print("Comparison:")
    print("  original obs[0:3] absolute max:", original_base_lin_vel_abs_max)
    print("  replay obs[0:3] absolute max:", float(np.max(np.abs(observations[:, 0:3]))))
    print("  actor command source: obs[9:12]")
    print("  logged command metadata max error:", float(np.max(command_metadata_error)))
    print(
        "  logged command metadata mismatched rows:",
        int(np.count_nonzero(np.max(command_metadata_error, axis=1) > 1e-6)),
    )
    print("  previous raw action max error (continuous rows):", previous_action_continuity_max)
    print("  simulator reset rows with zero previous action:", int(np.count_nonzero(reset_rows)))
    print("  runtime command-limit max adjustment:", float(np.max(runtime_command_clip_error)))
    print("  policy replay action max error:", float(np.max(action_error)))
    print("  policy replay action mean error:", float(np.mean(action_error)))
    print(
        "  policy vs motor-log action max error (non-reset rows):",
        float(np.nanmax(motor_action_error[non_reset_rows])),
    )
    print("  deployment vs sim q_target max error:", float(np.nanmax(target_error)))
    walking_rows = np.max(np.abs(commands), axis=1) > 0.02
    print(
        "  policy-limited target vs sim during walking max error:",
        float(np.nanmax(final_target_error[walking_rows])),
    )
    print(
        "  policy target clips (all rows / walking rows):",
        int(np.count_nonzero(np.abs(final_targets - replay_targets) > 1e-7)),
        "/",
        int(np.count_nonzero(np.abs(final_targets[walking_rows] - replay_targets[walking_rows]) > 1e-7)),
    )
    print(
        "  simulated measured-q max outside deployment bounds:",
        float(np.nanmax(sim_position_limit_error)),
    )
    print(
        "  simulated measured-q samples outside bounds by >0.01 rad:",
        int(np.count_nonzero(sim_position_limit_error > 0.01)),
    )
    print("  simulation default_q max abs:", float(np.nanmax(np.abs(sim_defaults))))
    print("  missing final motor rows:", int(np.count_nonzero(~np.isfinite(sim_targets))))
    print("Deployment shaping (always-policy dry replay):")
    print("  configured action clip:", action_clip)
    print("  configured hip action clip:", hip_action_clip)
    print("  configured action smoothing:", action_smoothing)
    print("  configured action delta limit:", action_delta_limit)
    print(
        "  logged action samples beyond clip:",
        int(np.count_nonzero(np.abs(sim_actions) > action_clip)) if action_clip > 0.0 else 0,
        "/",
        int(sim_actions.size),
    )
    print("  action shaping max/mean error:", float(np.max(action_shaping_error)), "/", float(np.mean(action_shaping_error)))
    print("  policy-limit adjusted samples:", int(np.count_nonzero(hard_limit_error > 1e-7)))
    print("  rate-limit adjusted samples:", int(np.count_nonzero(rate_limit_error > 1e-7)))
    print("  rate-limit max/mean target error:", float(np.max(rate_limit_error)), "/", float(np.mean(rate_limit_error)))
    print("  full runtime-vs-sim target max/mean error:", float(np.nanmax(total_runtime_target_error)), "/", float(np.nanmean(total_runtime_target_error)))
    print("  policy-to-motor-to-policy sign roundtrip max error:", float(np.max(sign_roundtrip_error)))
    print("  joint/motor torque sign roundtrip max error:", float(np.nanmax(torque_sign_roundtrip_error)))
    print(
        "  motor directions:",
        ", ".join(
            f"{name}={int(directions[index]):+d}"
            for index, name in enumerate(runner.policy_order)
        ),
    )
    print("Simulation-match path on logged physical state:")
    policy_torque_limits = [
        motor_layer.policy_pd_torque_limit_for_joint(joint_name)
        for joint_name in runner.policy_order
    ]
    print(
        "  estimated PD torque limits:",
        f"min={min(policy_torque_limits):.1f}",
        f"max={max(policy_torque_limits):.1f}",
    )
    print(
        "  sent-vs-sim target max/mean error:",
        float(np.nanmax(np.abs(sim_match_targets - sim_targets))),
        "/",
        float(np.nanmean(np.abs(sim_match_targets - sim_targets))),
    )
    print(
        "  estimated-vs-sim applied torque max/mean error:",
        float(np.nanmax(np.abs(sim_match_torque_estimate - applied_torques))),
        "/",
        float(np.nanmean(np.abs(sim_match_torque_estimate - applied_torques))),
    )
    print(
        "  estimated torque absolute max:",
        float(np.nanmax(np.abs(sim_match_torque_estimate))),
    )
    print("Simulation actuator identification and deployment wire gains:")
    torque_saturated = np.abs(computed_torques - applied_torques) > 1e-5
    for joint_index, joint_name in enumerate(runner.policy_order):
        kp_fit, kd_fit, fit_r2 = gain_fits[joint_index]
        group = joint_group(joint_name)
        configured_kp, configured_kd = motor_layer._joint_gains(
            "policy",
            joint_name,
            group,
        )
        effective_kp = motor_layer._effective_unsigned_wire_value(
            configured_kp,
            "kp",
            policy_command_proto,
        )
        effective_kd = motor_layer._effective_unsigned_wire_value(
            configured_kd,
            "kd",
            policy_command_proto,
        )
        torque_abs = np.abs(applied_torques[:, joint_index])
        print(
            f"  {joint_name:16s} sim_fit_kp/kd={kp_fit:6.1f}/{kd_fit:5.2f} "
            f"R2={fit_r2:.3f} deploy_cfg={configured_kp:.1f}/{configured_kd:.2f} "
            f"official_wire={effective_kp:.1f}/{effective_kd:.2f} "
            f"tau_p95/max={np.nanquantile(torque_abs, 0.95):.1f}/"
            f"{np.nanmax(torque_abs):.1f} "
            f"sim_saturation={np.nanmean(torque_saturated[:, joint_index]):.1%}"
        )
    policy_steps = {int(row["step"]) for row in rows}
    extra_motor_rows = sum(
        count
        for step, count in motor_row_count_by_step.items()
        if step not in policy_steps
    )
    print("  motor rows without a policy-input row:", int(extra_motor_rows))

    tolerance = 1e-5
    passed = (
        np.max(np.abs(observations[:, 0:3])) == 0.0
        and previous_action_continuity_max <= tolerance
        and np.max(runtime_command_clip_error) <= tolerance
        and np.max(action_error) <= tolerance
        and np.nanmax(motor_action_error[non_reset_rows]) <= tolerance
        and np.nanmax(target_error) <= tolerance
        and np.nanmax(sim_position_limit_error) <= 0.01
        and np.nanmax(np.abs(sim_defaults - runner.q_default[None, :])) <= tolerance
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "step", "sim_time", "command_label", "command_vx", "command_vy", "command_yaw",
            "action_max_error", "target_max_error",
        ]
        for joint_name in runner.policy_order:
            fieldnames.extend([
                f"sim_action__{joint_name}",
                f"replay_action__{joint_name}",
                f"sim_target__{joint_name}",
                f"replay_target__{joint_name}",
                f"limited_target__{joint_name}",
                f"runtime_action__{joint_name}",
                f"runtime_target__{joint_name}",
                f"motor_target__{joint_name}",
                f"sim_match_target__{joint_name}",
                f"sim_match_tau_est__{joint_name}",
                f"sim_q__{joint_name}",
            ])
        with output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                record = {
                    "step": row["step"],
                    "sim_time": row["sim_time"],
                    "command_label": command_label(commands[index]),
                    "command_vx": float(commands[index, 0]),
                    "command_vy": float(commands[index, 1]),
                    "command_yaw": float(commands[index, 2]),
                    "action_max_error": float(np.max(action_error[index])),
                    "target_max_error": float(np.nanmax(target_error[index]))
                    if np.any(np.isfinite(target_error[index])) else "",
                }
                for joint_index, joint_name in enumerate(runner.policy_order):
                    record[f"sim_action__{joint_name}"] = float(sim_actions[index, joint_index])
                    record[f"replay_action__{joint_name}"] = float(replay_actions[index, joint_index])
                    record[f"sim_target__{joint_name}"] = float(sim_targets[index, joint_index])
                    record[f"replay_target__{joint_name}"] = float(replay_targets[index, joint_index])
                    record[f"limited_target__{joint_name}"] = float(final_targets[index, joint_index])
                    record[f"runtime_action__{joint_name}"] = float(shaped_logged_actions[index, joint_index])
                    record[f"runtime_target__{joint_name}"] = float(runtime_targets[index, joint_index])
                    record[f"motor_target__{joint_name}"] = float(motor_targets[index, joint_index])
                    record[f"sim_match_target__{joint_name}"] = float(sim_match_targets[index, joint_index])
                    record[f"sim_match_tau_est__{joint_name}"] = float(sim_match_torque_estimate[index, joint_index])
                    record[f"sim_q__{joint_name}"] = float(sim_positions[index, joint_index])
                writer.writerow(record)
        print("Replay CSV:", output_path)

    if not passed and not args.allow_policy_mismatch:
        print("SIMULATION POLICY REPLAY COMPARISON FAILED")
        return 1
    if passed:
        print("SIMULATION POLICY REPLAY COMPARISON OK")
    else:
        print("SIMULATION POLICY REPLAY COMPARISON HAS MISMATCHES (allowed)")
    if np.any(np.abs(final_targets - replay_targets) > 1e-7):
        print("NOTE: policy/safety clipping is reported above but is not a policy-contract failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
