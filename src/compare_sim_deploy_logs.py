#!/usr/bin/env python3
"""Compare IsaacLab policy/motor CSV logs with the deployment policy and limits."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor


ROOT = Path(__file__).resolve().parents[1]


def load_csv(path):
    with Path(path).open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def vector(row, prefix, count, digits):
    return np.array(
        [float(row[f"{prefix}{index:0{digits}d}"]) for index in range(count)],
        dtype=np.float32,
    )


def command_vector(row):
    return np.array(
        [float(row["command_vx"]), float(row["command_vy"]), float(row["command_yaw"])],
        dtype=np.float32,
    )


def imu_vectors(row):
    angular_velocity = np.array(
        [row["base_ang_vel_x"], row["base_ang_vel_y"], row["base_ang_vel_z"]],
        dtype=np.float32,
    )
    gravity = np.array(
        [row["projected_gravity_x"], row["projected_gravity_y"], row["projected_gravity_z"]],
        dtype=np.float32,
    )
    return angular_velocity, gravity


def segment_summary(rows):
    segments = []
    start = 0
    for index in range(1, len(rows) + 1):
        changed = index == len(rows) or (
            rows[index]["command_segment_id"],
            rows[index]["command_label"],
        ) != (
            rows[start]["command_segment_id"],
            rows[start]["command_label"],
        )
        if not changed:
            continue
        first = rows[start]
        last = rows[index - 1]
        segments.append(
            {
                "id": first["command_segment_id"],
                "label": first["command_label"],
                "first_step": int(first["step"]),
                "last_step": int(last["step"]),
                "rows": index - start,
                "duration_s": (index - start) * 0.02,
                "logged_hold_s": float(last["command_hold_time_s"]),
                "vx": float(first["command_vx"]),
                "vy": float(first["command_vy"]),
                "yaw": float(first["command_yaw"]),
            }
        )
        start = index
    return segments


def reconstruct_observation(policy_row, joint_rows, previous_action):
    angular_velocity, gravity = imu_vectors(policy_row)
    command = command_vector(policy_row)
    q_relative = np.array(
        [float(row["q"]) - float(row["default_q"]) for row in joint_rows],
        dtype=np.float32,
    )
    qd = np.array([float(row["qd"]) for row in joint_rows], dtype=np.float32)
    return np.concatenate(
        [
            np.zeros(3, dtype=np.float32),
            angular_velocity,
            gravity,
            command,
            q_relative,
            qd,
            previous_action,
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_log")
    parser.add_argument("motor_log")
    parser.add_argument("--output", default=None)
    parser.add_argument("--walk-threshold", type=float, default=0.02)
    args = parser.parse_args()

    policy_rows = load_csv(args.policy_log)
    motor_rows = load_csv(args.motor_log)
    if not policy_rows or not motor_rows:
        raise ValueError("Both CSV logs must contain data rows")

    runner = PolicyRunner()
    safety = SafetyMonitor(runner.policy_order)
    motor_by_step = defaultdict(list)
    for row in motor_rows:
        motor_by_step[int(row["step"])].append(row)
    for rows in motor_by_step.values():
        rows.sort(key=lambda row: int(row["joint_index"]))

    first_joint_rows = motor_by_step[int(policy_rows[0]["step"])]
    sim_joint_order = [row["joint_name"] for row in first_joint_rows]
    order_matches = sim_joint_order == list(runner.policy_order)

    output_path = Path(args.output) if args.output else (
        ROOT / "logs" / f"deploy_dry_comparison_{Path(args.policy_log).stem}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exact_previous_action = np.zeros(12, dtype=np.float32)
    sim_target_previous = runner.q_default.copy()

    dry_q = runner.q_default.copy()
    dry_qd = np.zeros(12, dtype=np.float32)
    dry_previous_raw_action = np.zeros(12, dtype=np.float32)
    dry_previous_target = runner.q_default.copy()

    always_q = runner.q_default.copy()
    always_qd = np.zeros(12, dtype=np.float32)
    always_previous_raw_action = np.zeros(12, dtype=np.float32)
    always_previous_target = runner.q_default.copy()

    comparison_rows = []
    exact_action_errors = []
    reconstruction_errors = []
    sim_target_safety_errors = []
    dry_action_errors_active = []
    always_action_errors = []
    motor_tracking_errors = []
    incomplete_motor_steps = []
    rate_clip_total = 0
    position_clip_total = 0

    for policy_row in policy_rows:
        step = int(policy_row["step"])
        command = command_vector(policy_row)
        logged_obs = vector(policy_row, "obs_", 48, 3)
        logged_action = vector(policy_row, "action_", 12, 2)

        replay_action = runner.infer_action(logged_obs)
        exact_action_error = np.abs(replay_action - logged_action)
        exact_action_errors.extend(exact_action_error.tolist())

        joint_rows = motor_by_step.get(step, [])
        reconstruction_max_error = np.nan
        tracking_mean_error = np.nan
        tracking_max_error = np.nan
        if len(joint_rows) == 12:
            reconstructed = reconstruct_observation(
                policy_row,
                joint_rows,
                exact_previous_action,
            )
            reconstruction_error = np.abs(reconstructed - logged_obs)
            reconstruction_errors.extend(reconstruction_error.tolist())
            reconstruction_max_error = float(np.max(reconstruction_error))

            q_sim = np.array([float(row["q"]) for row in joint_rows], dtype=np.float32)
            q_target_sim = np.array(
                [float(row["q_target_est"]) for row in joint_rows],
                dtype=np.float32,
            )
            tracking_error = np.abs(q_target_sim - q_sim)
            motor_tracking_errors.extend(tracking_error.tolist())
            tracking_mean_error = float(np.mean(tracking_error))
            tracking_max_error = float(np.max(tracking_error))
        else:
            incomplete_motor_steps.append((step, len(joint_rows)))

        sim_q_target = runner.action_to_q_target(logged_action)
        hard_target = np.clip(sim_q_target, safety.q_min, safety.q_max)
        safe_sim_target = safety.safety_filter(sim_q_target, sim_target_previous)
        position_clip_count = int(np.count_nonzero(np.abs(hard_target - sim_q_target) > 1e-7))
        rate_clip_count = int(np.count_nonzero(np.abs(safe_sim_target - hard_target) > 1e-7))
        position_clip_total += position_clip_count
        rate_clip_total += rate_clip_count
        safety_error = np.abs(safe_sim_target - sim_q_target)
        sim_target_safety_errors.extend(safety_error.tolist())
        sim_target_previous = safe_sim_target

        walk_requested = bool(np.max(np.abs(command)) > float(args.walk_threshold))
        if walk_requested:
            dry_obs = runner.build_observation(
                base_ang_vel_b=np.zeros(3, dtype=np.float32),
                projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                command=command,
                q_current=dry_q,
                qd_current=dry_qd,
                previous_action=dry_previous_raw_action,
            )
            dry_action = runner.infer_action(dry_obs)
            dry_requested_target = runner.action_to_q_target(dry_action)
            dry_target = safety.safety_filter(dry_requested_target, dry_previous_target)
            dry_action_errors_active.extend(np.abs(dry_action - logged_action).tolist())
            dry_previous_raw_action = dry_action.copy()
        else:
            dry_action = np.zeros(12, dtype=np.float32)
            dry_target = dry_previous_target.copy()
            dry_previous_raw_action = np.zeros(12, dtype=np.float32)
        dry_qd = (dry_target - dry_q) / runner.control_dt
        dry_q = dry_target.copy()
        dry_previous_target = dry_target.copy()

        always_obs = runner.build_observation(
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            command=command,
            q_current=always_q,
            qd_current=always_qd,
            previous_action=always_previous_raw_action,
        )
        always_action = runner.infer_action(always_obs)
        always_requested_target = runner.action_to_q_target(always_action)
        always_target = safety.safety_filter(always_requested_target, always_previous_target)
        always_action_errors.extend(np.abs(always_action - logged_action).tolist())
        always_qd = (always_target - always_q) / runner.control_dt
        always_q = always_target.copy()
        always_previous_target = always_target.copy()
        always_previous_raw_action = always_action.copy()

        comparison_rows.append(
            {
                "step": step,
                "sim_time": float(policy_row["sim_time"]),
                "command_label": policy_row["command_label"],
                "command_vx": float(command[0]),
                "command_vy": float(command[1]),
                "command_yaw": float(command[2]),
                "exact_action_max_error": float(np.max(exact_action_error)),
                "obs_reconstruction_max_error": reconstruction_max_error,
                "sim_action_abs_max": float(np.max(np.abs(logged_action))),
                "main_dry_action_abs_max": float(np.max(np.abs(dry_action))),
                "always_policy_dry_action_abs_max": float(np.max(np.abs(always_action))),
                "sim_target_safety_mean_error": float(np.mean(safety_error)),
                "sim_target_safety_max_error": float(np.max(safety_error)),
                "position_clip_count": position_clip_count,
                "rate_clip_count": rate_clip_count,
                "sim_tracking_mean_error": tracking_mean_error,
                "sim_tracking_max_error": tracking_max_error,
            }
        )
        exact_previous_action = logged_action.copy()

    fieldnames = list(comparison_rows[0])
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    def mean_max(values):
        if not values:
            return float("nan"), float("nan")
        array = np.asarray(values, dtype=np.float64)
        return float(np.mean(array)), float(np.max(array))

    exact_mean, exact_max = mean_max(exact_action_errors)
    reconstruction_mean, reconstruction_max = mean_max(reconstruction_errors)
    safety_mean, safety_max = mean_max(sim_target_safety_errors)
    dry_mean, dry_max = mean_max(dry_action_errors_active)
    always_mean, always_max = mean_max(always_action_errors)
    tracking_mean, tracking_max = mean_max(motor_tracking_errors)

    print("Policy:", runner.policy_path)
    print("Policy SHA256:", runner.policy_sha256)
    print("Policy rows:", len(policy_rows))
    print("Motor rows:", len(motor_rows))
    print("Simulation joint order:", ", ".join(sim_joint_order))
    print("Deployment joint order:", ", ".join(runner.policy_order))
    print("Joint order matches:", order_matches)
    print("Segments:")
    for segment in segment_summary(policy_rows):
        print(
            f"  {segment['label']:8s} steps={segment['first_step']:03d}-{segment['last_step']:03d} "
            f"rows={segment['rows']:3d} duration={segment['duration_s']:.2f}s "
            f"cmd=[{segment['vx']:+.2f},{segment['vy']:+.2f},{segment['yaw']:+.2f}]"
        )
    print(f"Exact observation replay action error: mean={exact_mean:.3e} max={exact_max:.3e}")
    print(
        "Observation reconstruction error: "
        f"mean={reconstruction_mean:.3e} max={reconstruction_max:.3e}"
    )
    print(
        "Configured safety distortion of simulation q targets: "
        f"mean={safety_mean:.4f} rad max={safety_max:.4f} rad"
    )
    print("Position clips:", position_clip_total, "Rate clips:", rate_clip_total)
    print(
        "Main-state-machine fake dry action error during movement: "
        f"mean={dry_mean:.4f} max={dry_max:.4f}"
    )
    print(
        "Always-policy fake dry action error: "
        f"mean={always_mean:.4f} max={always_max:.4f}"
    )
    print(
        "Simulation motor tracking |q_target-q|: "
        f"mean={tracking_mean:.4f} rad max={tracking_max:.4f} rad"
    )
    if incomplete_motor_steps:
        print("Incomplete motor-log steps:", incomplete_motor_steps)
    print("Comparison CSV:", output_path)
    return 0 if order_matches and exact_max < 1e-5 and reconstruction_max < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
