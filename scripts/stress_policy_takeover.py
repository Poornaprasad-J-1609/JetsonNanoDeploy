#!/usr/bin/env python3
"""Replay real hardware states through randomized stand/policy transitions."""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from can_topology import resolve_joint_can_bus
from main_controller import (
    filtered_policy_action,
    policy_entry_gain_blend_scale,
    shifted_safety_filter,
    smoothstep,
    stand_recovery_gain_blend_scale,
)
from motor_command_layer import MotorCommandLayer
from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor


COMMANDS = (
    ("forward", (0.20, 0.00, 0.00)),
    ("backward", (-0.20, 0.00, 0.00)),
    ("left", (0.00, 0.12, 0.00)),
    ("right", (0.00, -0.12, 0.00)),
    ("forward_left", (0.18, 0.10, 0.00)),
    ("forward_right", (0.18, -0.10, 0.00)),
    ("backward_left", (-0.18, 0.10, 0.00)),
    ("backward_right", (-0.18, -0.10, 0.00)),
    ("yaw_left", (0.00, 0.00, 0.12)),
    ("yaw_right", (0.00, 0.00, -0.12)),
    ("forward_yaw_left", (0.13, 0.00, 0.10)),
    ("backward_yaw_right", (-0.13, 0.00, -0.10)),
)


def _finite_vector(row, prefix, count):
    values = []
    for index in range(count):
        text = row.get(f"{prefix}{index:02d}", "")
        if text in ("", None):
            return None
        value = float(text)
        if not np.isfinite(value):
            return None
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _finite_observation(row):
    values = []
    for index in range(48):
        text = row.get(f"obs_{index:03d}", "")
        if text in ("", None):
            return None
        value = float(text)
        if not np.isfinite(value):
            return None
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def load_real_policy_samples(paths):
    samples = []
    for path in paths:
        path = Path(path).expanduser().resolve()
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                obs = _finite_observation(row)
                q = _finite_vector(row, "q_", 12)
                qd = _finite_vector(row, "qd_", 12)
                if obs is None or q is None or qd is None:
                    continue
                entry_text = row.get("policy_entry_scale", "")
                if entry_text not in ("", None):
                    entry_scale = float(entry_text)
                    if not np.isfinite(entry_scale) or entry_scale > 0.02:
                        continue
                # A takeover starts only after the real stand-readiness gate.
                # Mid-gait rows remain valuable for gait analysis, but using
                # one as a synthetic stand state creates a false target jump.
                if float(np.max(np.abs(q))) > 0.08:
                    continue
                gravity_norm = float(np.linalg.norm(obs[6:9]))
                if gravity_norm < 0.5 or gravity_norm > 1.5:
                    continue
                samples.append(
                    {
                        "source": str(path),
                        "step": row.get("step", ""),
                        "gyro": obs[3:6].copy(),
                        "gravity": (obs[6:9] / gravity_norm).astype(np.float32),
                        "q": q,
                        "qd": qd,
                        "previous_action": obs[36:48].copy(),
                    }
                )
    if not samples:
        raise ValueError(
            "no complete real policy rows were found; logs need obs_000..047, "
            "q_00..11, and qd_00..11"
        )
    return samples


def feedback_snapshot(layer, q, qd):
    now = time.monotonic()
    feedback = {}
    for index, joint_name in enumerate(layer.policy_order):
        direction = float(layer.joint_directions[joint_name])
        offset = float(layer.joint_offsets[joint_name])
        motor_position = offset + direction * float(q[index])
        motor_velocity = direction * float(qd[index])
        feedback[joint_name] = {
            "timestamp": now,
            "fault_bits": 0,
            "joint_position": float(q[index]),
            "joint_velocity": float(qd[index]),
            "position": motor_position,
            "velocity": motor_velocity,
            "position_raw": motor_position,
            "velocity_raw": motor_velocity,
            "torque": 0.0,
            "torque_raw": 0.0,
        }
    return feedback


def command_arrays(layer, commands, fallback):
    q_sent = np.asarray(fallback, dtype=np.float32).copy()
    tau = np.zeros(len(layer.policy_order), dtype=np.float32)
    kp = np.zeros(len(layer.policy_order), dtype=np.float32)
    kd = np.zeros(len(layer.policy_order), dtype=np.float32)
    for command in commands:
        index = layer.policy_index_by_joint[command["joint_name"]]
        q_sent[index] = float(command["q_des"])
        tau[index] = float(command["tau_pd_est"])
        kp[index] = float(command["kp_effective"])
        kd[index] = float(command["kd_effective"])
    return q_sent, tau, kp, kd


def run_case(case_index, sample, command_name, command, runner, layer, safety, rng):
    dt = float(runner.control_dt)
    policy_torque_limit = max(
        layer.policy_pd_torque_limit_for_joint(name)
        for name in layer.policy_order
    )
    entry_steps = int(round(2.0 / dt))
    steady_steps = int(round(1.5 / dt))
    return_steps = int(round(2.0 / dt))
    plant_response = float(rng.uniform(0.08, 0.24))
    gyro_noise_std = float(rng.uniform(0.0, 0.05))
    gravity_noise_std = float(rng.uniform(0.0, 0.02))
    q_noise_std = float(rng.uniform(0.0, 0.0015))
    qd_noise_std = float(rng.uniform(0.0, 0.05))
    process_noise_std = float(rng.uniform(0.0, 0.0005))
    load_scale = float(rng.uniform(0.85, 1.15))

    q = np.clip(sample["q"], safety.q_min, safety.q_max).astype(np.float32)
    qd = np.asarray(sample["qd"], dtype=np.float32).copy()
    previous_raw_action = np.asarray(
        sample["previous_action"], dtype=np.float32
    ).copy()
    previous_sent_action = np.zeros(12, dtype=np.float32)
    q_previous_target = runner.q_stand.astype(np.float32).copy()

    initial_feedback = feedback_snapshot(layer, q, qd)
    stand_commands = layer.build_mit_commands(
        runner.q_stand,
        phase="stand",
        feedback_by_joint=initial_feedback,
    )
    previous_q_sent, previous_tau, previous_kp, previous_kd = command_arrays(
        layer, stand_commands, runner.q_stand
    )
    # At the measured stand equilibrium, motor PD torque balances body load.
    # Preserve that signed load while policy gains and targets change so the
    # dry plant can expose support loss instead of assuming a weightless body.
    external_load_torque = (-previous_tau * load_scale).astype(np.float32)

    maximum_target_step = 0.0
    maximum_torque = 0.0
    maximum_policy_torque = 0.0
    maximum_return_torque = 0.0
    maximum_torque_step = 0.0
    maximum_torque_step_at = ""
    maximum_torque_step_joint = ""
    maximum_torque_step_before = 0.0
    maximum_torque_step_after = 0.0
    maximum_kp_step = 0.0
    maximum_kd_step = 0.0
    maximum_policy_tracking_error = 0.0
    maximum_steady_tracking_error = 0.0
    terminal_policy_tracking_error = 0.0
    maximum_policy_speed = 0.0
    maximum_policy_load_deficit = 0.0
    terminal_policy_load_deficit = 0.0
    maximum_raw_action = 0.0
    maximum_sent_action = 0.0
    maximum_requested_target_excursion = 0.0
    maximum_sent_target_excursion = 0.0
    steady_sent_targets = []
    steady_policy_torques = []
    takeover_target_step = None
    takeover_torque_step = None
    return_target_step = None
    return_torque_step = None
    nonfinite = False
    torque_limit_violation = False
    hard_limit_violation = False
    command_rate_violation = False

    total_policy_steps = entry_steps + steady_steps
    last_policy_gain_alpha = 0.0
    return_start = None
    for step in range(total_policy_steps + return_steps):
        q_measured = q + rng.normal(0.0, q_noise_std, 12).astype(np.float32)
        qd_measured = qd + rng.normal(0.0, qd_noise_std, 12).astype(np.float32)
        q_measured = np.clip(q_measured, safety.q_min, safety.q_max)
        feedback = feedback_snapshot(layer, q_measured, qd_measured)

        if step < total_policy_steps:
            alpha = smoothstep(min(1.0, float(step + 1) / entry_steps))
            noisy_gyro = (
                sample["gyro"]
                + rng.normal(0.0, gyro_noise_std, 3).astype(np.float32)
            )
            noisy_gravity = (
                sample["gravity"]
                + rng.normal(0.0, gravity_noise_std, 3).astype(np.float32)
            )
            noisy_gravity /= max(float(np.linalg.norm(noisy_gravity)), 1.0e-6)
            obs = runner.build_observation(
                base_ang_vel_b=noisy_gyro,
                projected_gravity_b=noisy_gravity,
                command=np.asarray(command, dtype=np.float32),
                q_current=q_measured,
                qd_current=qd_measured,
                previous_action=previous_raw_action,
            )
            raw_action = runner.infer_action(obs)
            sent_action = filtered_policy_action(
                raw_action=raw_action,
                previous_action=previous_sent_action,
                clip_abs=3.2,
                smoothing=0.35,
                delta_limit_abs=0.20,
            )
            sent_action = (sent_action * float(alpha)).astype(np.float32)
            q_requested = runner.action_to_q_target(sent_action)
            maximum_raw_action = max(
                maximum_raw_action,
                float(np.max(np.abs(raw_action))),
            )
            maximum_sent_action = max(
                maximum_sent_action,
                float(np.max(np.abs(sent_action))),
            )
            maximum_requested_target_excursion = max(
                maximum_requested_target_excursion,
                float(np.max(np.abs(q_requested - runner.q_stand))),
            )
            q_safe = shifted_safety_filter(
                safety,
                q_requested,
                q_previous_target,
                np.zeros(12, dtype=np.float32),
                apply_rate_limit=True,
                use_policy_limits=False,
            )
            last_policy_gain_alpha = policy_entry_gain_blend_scale(
                float(step + 1) * dt,
                2.0,
            )
            commands = layer.build_mit_commands(
                q_safe,
                phase="policy",
                feedback_by_joint=feedback,
                prelimit_q_target=q_requested,
                gain_blend_from_phase="stand",
                gain_blend_alpha=last_policy_gain_alpha,
                previous_command_q=q_previous_target,
                max_command_delta=safety.dq_max,
            )
            previous_raw_action = raw_action.astype(np.float32)
            previous_sent_action = sent_action
        else:
            return_index = step - total_policy_steps + 1
            if return_start is None:
                return_start = q_previous_target.copy()
            return_alpha = smoothstep(min(1.0, return_index / return_steps))
            q_requested = (
                (1.0 - return_alpha) * return_start
                + return_alpha * runner.q_stand
            ).astype(np.float32)
            q_safe = shifted_safety_filter(
                safety,
                q_requested,
                q_previous_target,
                np.zeros(12, dtype=np.float32),
                apply_rate_limit=True,
                use_policy_limits=False,
            )
            commands = layer.build_mit_commands(
                q_safe,
                phase="stand",
                feedback_by_joint=feedback,
                gain_blend_from_phase="policy",
                gain_blend_alpha=stand_recovery_gain_blend_scale(
                    last_policy_gain_alpha,
                    float(return_index) * dt,
                    2.0,
                ),
            )

        q_sent, tau, kp, kd = command_arrays(layer, commands, q_safe)
        target_step = float(np.max(np.abs(q_sent - previous_q_sent)))
        torque_step = float(np.max(np.abs(tau - previous_tau)))
        kp_step = float(np.max(np.abs(kp - previous_kp)))
        kd_step = float(np.max(np.abs(kd - previous_kd)))
        maximum_target_step = max(maximum_target_step, target_step)
        maximum_torque = max(maximum_torque, float(np.max(np.abs(tau))))
        if step < total_policy_steps:
            maximum_policy_torque = max(
                maximum_policy_torque,
                float(np.max(np.abs(tau))),
            )
        else:
            maximum_return_torque = max(
                maximum_return_torque,
                float(np.max(np.abs(tau))),
            )
        if torque_step > maximum_torque_step:
            torque_step_index = int(np.argmax(np.abs(tau - previous_tau)))
            maximum_torque_step = torque_step
            maximum_torque_step_at = (
                f"policy:{step}"
                if step < total_policy_steps
                else f"return:{step - total_policy_steps}"
            )
            maximum_torque_step_joint = runner.policy_order[torque_step_index]
            maximum_torque_step_before = float(previous_tau[torque_step_index])
            maximum_torque_step_after = float(tau[torque_step_index])
        maximum_kp_step = max(maximum_kp_step, kp_step)
        maximum_kd_step = max(maximum_kd_step, kd_step)
        tracking_error = float(np.max(np.abs(q_sent - q_measured)))
        load_deficit = float(np.max(np.abs(tau + external_load_torque)))
        if step < total_policy_steps:
            maximum_sent_target_excursion = max(
                maximum_sent_target_excursion,
                float(np.max(np.abs(q_sent - runner.q_stand))),
            )
            if step >= entry_steps:
                steady_sent_targets.append(q_sent.copy())
                steady_policy_torques.append(tau.copy())
            maximum_policy_tracking_error = max(
                maximum_policy_tracking_error,
                tracking_error,
            )
            maximum_policy_speed = max(
                maximum_policy_speed,
                float(np.max(np.abs(qd_measured))),
            )
            maximum_policy_load_deficit = max(
                maximum_policy_load_deficit,
                load_deficit,
            )
            if step >= entry_steps:
                maximum_steady_tracking_error = max(
                    maximum_steady_tracking_error,
                    tracking_error,
                )
            if step == total_policy_steps - 1:
                terminal_policy_tracking_error = tracking_error
                terminal_policy_load_deficit = load_deficit
        if step == 0:
            takeover_target_step = target_step
            takeover_torque_step = torque_step
        if step == total_policy_steps:
            return_target_step = target_step
            return_torque_step = torque_step

        values = np.concatenate((q_sent, tau, kp, kd))
        nonfinite = nonfinite or not bool(np.all(np.isfinite(values)))
        if step < total_policy_steps:
            torque_limit_violation = torque_limit_violation or bool(
                np.max(np.abs(tau)) > policy_torque_limit + 0.05
            )
        hard_limit_violation = hard_limit_violation or bool(
            np.any(q_sent < safety.q_min - 1.0e-6)
            or np.any(q_sent > safety.q_max + 1.0e-6)
        )
        command_rate_violation = command_rate_violation or bool(
            np.any(np.abs(q_sent - previous_q_sent) > safety.dq_max + 1.0e-5)
        )

        # First-order loaded-joint model. Net motor-plus-body torque changes
        # joint position relative to the commanded impedance. It is deliberately
        # conservative: policy torque clipping cannot hide a support deficit.
        net_torque = tau + external_load_torque
        effective_stiffness = np.maximum(np.abs(kp), 20.0)
        q_next = (
            q
            + plant_response * net_torque / effective_stiffness
            + rng.normal(0.0, process_noise_std, 12).astype(np.float32)
        )
        q_next = np.clip(q_next, safety.q_min, safety.q_max).astype(np.float32)
        qd = ((q_next - q) / dt).astype(np.float32)
        q = q_next
        q_previous_target = q_sent
        previous_q_sent = q_sent
        previous_tau = tau
        previous_kp = kp
        previous_kd = kd

    if steady_sent_targets:
        steady_target_range = np.ptp(
            np.asarray(steady_sent_targets, dtype=np.float32),
            axis=0,
        )
    else:
        steady_target_range = np.zeros(12, dtype=np.float32)
    if steady_policy_torques:
        steady_torque_range = np.ptp(
            np.asarray(steady_policy_torques, dtype=np.float32),
            axis=0,
        )
    else:
        steady_torque_range = np.zeros(12, dtype=np.float32)
    sagittal_indices = [
        index
        for index, name in enumerate(runner.policy_order)
        if "_thigh_" in name or "_calf_" in name
    ]
    sagittal_ranges = steady_target_range[sagittal_indices]
    common_pass = (
        not nonfinite
        and not torque_limit_violation
        and not hard_limit_violation
        and not command_rate_violation
    )
    takeover_passed = (
        common_pass
        and float(takeover_target_step) <= 0.02
        and float(takeover_torque_step) <= 8.0
    )
    # The actor can legitimately reverse a bounded policy torque between
    # mid-gait samples. Return remains a separate result because the loaded
    # first-order plant can move appreciably during the cycle before release.
    return_passed = (
        common_pass
        and float(return_target_step) <= 0.02
        and float(return_torque_step) <= 12.0
        # The loaded legacy stand path reached about 71 Nm transiently in the
        # real logs. Keep synthetic noisy recovery below a 90 Nm diagnostic
        # ceiling while leaving the proven pose packet behavior unchanged.
        and maximum_return_torque <= 90.0
        and maximum_kp_step <= 21.0
        and maximum_kd_step <= 1.10
    )
    passed = takeover_passed and return_passed
    result = {
        "case": case_index,
        "passed": int(passed),
        "takeover_passed": int(takeover_passed),
        "return_passed": int(return_passed),
        "source_log": sample["source"],
        "source_step": sample["step"],
        "command": command_name,
        "command_vx": command[0],
        "command_vy": command[1],
        "command_yaw": command[2],
        "plant_response": plant_response,
        "load_scale": load_scale,
        "external_load_torque_abs_max_nm": float(
            np.max(np.abs(external_load_torque))
        ),
        "gyro_noise_std": gyro_noise_std,
        "gravity_noise_std": gravity_noise_std,
        "q_noise_std": q_noise_std,
        "qd_noise_std": qd_noise_std,
        "maximum_target_step_rad": maximum_target_step,
        "takeover_target_step_rad": takeover_target_step,
        "return_target_step_rad": return_target_step,
        "maximum_abs_torque_nm": maximum_torque,
        "maximum_policy_torque_nm": maximum_policy_torque,
        "maximum_return_torque_nm": maximum_return_torque,
        "maximum_torque_step_nm": maximum_torque_step,
        "maximum_torque_step_at": maximum_torque_step_at,
        "maximum_torque_step_joint": maximum_torque_step_joint,
        "maximum_torque_step_before_nm": maximum_torque_step_before,
        "maximum_torque_step_after_nm": maximum_torque_step_after,
        "takeover_torque_step_nm": takeover_torque_step,
        "return_torque_step_nm": return_torque_step,
        "maximum_kp_step": maximum_kp_step,
        "maximum_kd_step": maximum_kd_step,
        "maximum_policy_tracking_error_rad": maximum_policy_tracking_error,
        "maximum_steady_tracking_error_rad": maximum_steady_tracking_error,
        "terminal_policy_tracking_error_rad": terminal_policy_tracking_error,
        "maximum_policy_speed_rad_s": maximum_policy_speed,
        "maximum_policy_load_deficit_nm": maximum_policy_load_deficit,
        "terminal_policy_load_deficit_nm": terminal_policy_load_deficit,
        "maximum_raw_action": maximum_raw_action,
        "maximum_sent_action": maximum_sent_action,
        "maximum_requested_target_excursion_rad": maximum_requested_target_excursion,
        "maximum_sent_target_excursion_rad": maximum_sent_target_excursion,
        "minimum_steady_sagittal_target_range_rad": float(
            np.min(sagittal_ranges)
        ),
        "mean_steady_sagittal_target_range_rad": float(
            np.mean(sagittal_ranges)
        ),
        "maximum_steady_sagittal_target_range_rad": float(
            np.max(sagittal_ranges)
        ),
        "minimum_steady_sagittal_torque_range_nm": float(
            np.min(steady_torque_range[sagittal_indices])
        ),
        "mean_steady_sagittal_torque_range_nm": float(
            np.mean(steady_torque_range[sagittal_indices])
        ),
        "maximum_steady_sagittal_torque_range_nm": float(
            np.max(steady_torque_range[sagittal_indices])
        ),
        "nonfinite": int(nonfinite),
        "torque_limit_violation": int(torque_limit_violation),
        "hard_limit_violation": int(hard_limit_violation),
        "command_rate_violation": int(command_rate_violation),
    }
    for index, joint_name in enumerate(runner.policy_order):
        result[f"{joint_name}_steady_target_range_rad"] = float(
            steady_target_range[index]
        )
        result[f"{joint_name}_steady_torque_range_nm"] = float(
            steady_torque_range[index]
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-log", action="append", required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8675)
    parser.add_argument("--policy-kp-scale", type=float, default=1.0)
    parser.add_argument("--policy-kd-scale", type=float, default=1.0)
    parser.add_argument("--policy-torque-limit", type=float, default=30.0)
    parser.add_argument(
        "--summary-csv",
        default=str(ROOT / "logs" / "policy_takeover_stress_summary.csv"),
    )
    parser.add_argument("--allow-policy-hash-mismatch", action="store_true")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be >= 1")
    if not np.isfinite(args.policy_kp_scale) or args.policy_kp_scale <= 0.0:
        parser.error("--policy-kp-scale must be finite and > 0")
    if not np.isfinite(args.policy_kd_scale) or args.policy_kd_scale <= 0.0:
        parser.error("--policy-kd-scale must be finite and > 0")
    if not np.isfinite(args.policy_torque_limit) or args.policy_torque_limit <= 0.0:
        parser.error("--policy-torque-limit must be finite and > 0")

    runner = PolicyRunner(
        allow_policy_hash_mismatch=args.allow_policy_hash_mismatch
    )
    samples = load_real_policy_samples(args.real_log)
    motor_ids_path = ROOT / "config" / "motor_ids.yaml"
    import yaml

    with motor_ids_path.open(encoding="utf-8") as stream:
        motor_ids = yaml.safe_load(stream)["motor_ids"]
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=[],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 2),
    )
    for group in ("hip", "thigh", "calf"):
        layer.gains["policy"][group]["kp"] *= float(args.policy_kp_scale)
        layer.gains["policy"][group]["kd"] *= float(args.policy_kd_scale)
    for joint_cfg in layer.gains["policy"].get("joints", {}).values():
        joint_cfg["kp"] *= float(args.policy_kp_scale)
        joint_cfg["kd"] *= float(args.policy_kd_scale)
    layer.set_policy_pd_torque_limit(args.policy_torque_limit)
    safety = SafetyMonitor(runner.policy_order, control_dt=runner.control_dt)
    rng = np.random.default_rng(args.seed)

    rows = []
    for case_index in range(args.cases):
        sample = samples[int(rng.integers(0, len(samples)))]
        command_name, command = COMMANDS[case_index % len(COMMANDS)]
        rows.append(
            run_case(
                case_index,
                sample,
                command_name,
                command,
                runner,
                layer,
                safety,
                rng,
            )
        )

    output = Path(args.summary_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(row["passed"] for row in rows)
    takeover_passed = sum(row["takeover_passed"] for row in rows)
    return_passed = sum(row["return_passed"] for row in rows)
    print("POLICY TAKEOVER STRESS SUMMARY")
    print(f"policy_sha256: {runner.policy_sha256}")
    print(f"policy_kp_scale: {args.policy_kp_scale:.3f}")
    print(f"policy_kd_scale: {args.policy_kd_scale:.3f}")
    print(f"policy_torque_limit_nm: {args.policy_torque_limit:.3f}")
    print(f"real_policy_samples: {len(samples)}")
    print(f"cases: {len(rows)}")
    print(f"takeover_passed: {takeover_passed}")
    print(f"takeover_failed: {len(rows) - takeover_passed}")
    print(f"return_passed: {return_passed}")
    print(f"return_failed: {len(rows) - return_passed}")
    print(f"passed: {passed}")
    print(f"failed: {len(rows) - passed}")
    for field in (
        "maximum_target_step_rad",
        "takeover_target_step_rad",
        "return_target_step_rad",
        "maximum_abs_torque_nm",
        "maximum_policy_torque_nm",
        "maximum_return_torque_nm",
        "maximum_torque_step_nm",
        "takeover_torque_step_nm",
        "return_torque_step_nm",
        "maximum_kp_step",
        "maximum_kd_step",
        "external_load_torque_abs_max_nm",
        "maximum_policy_tracking_error_rad",
        "maximum_steady_tracking_error_rad",
        "terminal_policy_tracking_error_rad",
        "maximum_policy_speed_rad_s",
        "maximum_policy_load_deficit_nm",
        "terminal_policy_load_deficit_nm",
        "maximum_raw_action",
        "maximum_sent_action",
        "maximum_requested_target_excursion_rad",
        "maximum_sent_target_excursion_rad",
        "minimum_steady_sagittal_target_range_rad",
        "mean_steady_sagittal_target_range_rad",
        "maximum_steady_sagittal_target_range_rad",
        "minimum_steady_sagittal_torque_range_nm",
        "mean_steady_sagittal_torque_range_nm",
        "maximum_steady_sagittal_torque_range_nm",
    ):
        print(f"{field}: {max(float(row[field]) for row in rows):.6f}")
    print(f"summary_csv: {output}")
    if passed != len(rows):
        failed_cases = [str(row["case"]) for row in rows if not row["passed"]]
        print("failed_cases:", ", ".join(failed_cases))
        return 1
    print("PASS: all randomized real-log takeover cases remained smooth and bounded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
