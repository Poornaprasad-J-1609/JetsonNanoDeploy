#!/usr/bin/env python3
"""Replay IsaacLab policy-input CSV rows and compare deployment outputs."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor


def floats(row, prefix, count, width):
    return np.asarray(
        [float(row[f"{prefix}_{index:0{width}d}"]) for index in range(count)],
        dtype=np.float32,
    )


def command_segments(rows):
    segments = []
    start = 0
    for index in range(1, len(rows) + 1):
        changed = index == len(rows)
        if not changed:
            current = rows[start]
            candidate = rows[index]
            changed = (
                candidate["command_segment_id"],
                candidate["command_label"],
            ) != (
                current["command_segment_id"],
                current["command_label"],
            )
        if changed:
            first = rows[start]
            last = rows[index - 1]
            segments.append({
                "id": first["command_segment_id"],
                "label": first["command_label"],
                "step_start": int(first["step"]),
                "step_end": int(last["step"]),
                "time_start": float(first["sim_time"]),
                "time_end": float(last["sim_time"]),
                "vx": float(first["command_vx"]),
                "vy": float(first["command_vy"]),
                "yaw": float(first["command_yaw"]),
                "rows": index - start,
            })
            start = index
    return segments


def load_motor_rows(path):
    by_step_joint = {}
    order = {}
    with Path(path).open(newline="") as motor_file:
        for row in csv.DictReader(motor_file):
            step = int(row["step"])
            joint_name = row["joint_name"]
            by_step_joint[(step, joint_name)] = row
            order[int(row["joint_index"])] = joint_name
    return by_step_joint, [order[index] for index in sorted(order)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_input_log")
    parser.add_argument("motor_log")
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    runner = PolicyRunner(policy_path=args.policy_path)
    safety = SafetyMonitor(runner.policy_order)
    with Path(args.policy_input_log).open(newline="") as policy_file:
        rows = list(csv.DictReader(policy_file))
    if not rows:
        raise ValueError("Policy input log has no rows")

    motor_rows, sim_joint_order = load_motor_rows(args.motor_log)
    if sim_joint_order != list(runner.policy_order):
        raise ValueError(
            "Joint order mismatch:\n"
            f"  simulation={sim_joint_order}\n"
            f"  deployment={list(runner.policy_order)}"
        )

    observations = np.stack([floats(row, "obs", 48, 3) for row in rows])
    sim_actions = np.stack([floats(row, "action", 12, 2) for row in rows])
    commands = np.asarray(
        [[float(row["command_vx"]), float(row["command_vy"]), float(row["command_yaw"])] for row in rows],
        dtype=np.float32,
    )

    with torch.no_grad():
        replay_actions = runner.policy(torch.from_numpy(observations)).cpu().numpy()
    replay_targets = runner.q_default[None, :] + runner.action_scale * replay_actions
    final_targets = np.clip(replay_targets, safety.q_min[None, :], safety.q_max[None, :])

    sim_targets = np.full_like(replay_targets, np.nan)
    sim_motor_actions = np.full_like(replay_actions, np.nan)
    sim_defaults = np.full_like(replay_targets, np.nan)
    for row_index, policy_row in enumerate(rows):
        step = int(policy_row["step"])
        for joint_index, joint_name in enumerate(runner.policy_order):
            motor_row = motor_rows.get((step, joint_name))
            if motor_row is None:
                continue
            sim_targets[row_index, joint_index] = float(motor_row["q_target_est"])
            sim_motor_actions[row_index, joint_index] = float(motor_row["action"])
            sim_defaults[row_index, joint_index] = float(motor_row["default_q"])

    action_error = np.abs(replay_actions - sim_actions)
    target_error = np.abs(replay_targets - sim_targets)
    final_target_error = np.abs(final_targets - sim_targets)
    motor_action_error = np.abs(sim_motor_actions - sim_actions)
    command_error = np.abs(observations[:, 9:12] - commands)
    previous_action_error = np.abs(observations[1:, 36:48] - sim_actions[:-1])

    print("Policy:", runner.policy_path)
    print("Policy SHA256:", runner.policy_sha256)
    print("Rows:", len(rows))
    print("Joint order:", ", ".join(sim_joint_order))
    print("Segments:")
    for segment in command_segments(rows):
        duration = segment["rows"] * runner.control_dt
        print(
            f"  {segment['id']:>2s} {segment['label']:9s} "
            f"steps={segment['step_start']:04d}..{segment['step_end']:04d} "
            f"duration={duration:.2f}s "
            f"cmd=[{segment['vx']:+.3f},{segment['vy']:+.3f},{segment['yaw']:+.3f}]"
        )

    print("Comparison:")
    print("  obs[0:3] absolute max:", float(np.max(np.abs(observations[:, 0:3]))))
    print("  command vs obs[9:12] max error:", float(np.max(command_error)))
    print("  previous raw action max error:", float(np.max(previous_action_error)))
    print("  policy replay action max error:", float(np.max(action_error)))
    print("  policy replay action mean error:", float(np.mean(action_error)))
    print("  policy vs motor-log action max error:", float(np.nanmax(motor_action_error)))
    print("  deployment vs sim q_target max error:", float(np.nanmax(target_error)))
    walking_rows = np.max(np.abs(commands), axis=1) > 0.02
    print(
        "  final limited target vs sim during walking max error:",
        float(np.nanmax(final_target_error[walking_rows])),
    )
    print(
        "  hard position clips (all rows / walking rows):",
        int(np.count_nonzero(np.abs(final_targets - replay_targets) > 1e-7)),
        "/",
        int(np.count_nonzero(np.abs(final_targets[walking_rows] - replay_targets[walking_rows]) > 1e-7)),
    )
    print("  simulation default_q max abs:", float(np.nanmax(np.abs(sim_defaults))))
    print("  missing final motor rows:", int(np.count_nonzero(~np.isfinite(sim_targets))))

    tolerance = 1e-5
    passed = (
        np.max(np.abs(observations[:, 0:3])) == 0.0
        and np.max(command_error) <= tolerance
        and np.max(previous_action_error) <= tolerance
        and np.max(action_error) <= tolerance
        and np.nanmax(motor_action_error) <= tolerance
        and np.nanmax(target_error) <= tolerance
        and np.nanmax(final_target_error[walking_rows]) <= tolerance
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
            ])
        with output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                record = {
                    "step": row["step"],
                    "sim_time": row["sim_time"],
                    "command_label": row["command_label"],
                    "command_vx": row["command_vx"],
                    "command_vy": row["command_vy"],
                    "command_yaw": row["command_yaw"],
                    "action_max_error": float(np.max(action_error[index])),
                    "target_max_error": float(np.nanmax(target_error[index]))
                    if np.any(np.isfinite(target_error[index])) else "",
                }
                for joint_index, joint_name in enumerate(runner.policy_order):
                    record[f"sim_action__{joint_name}"] = float(sim_actions[index, joint_index])
                    record[f"replay_action__{joint_name}"] = float(replay_actions[index, joint_index])
                    record[f"sim_target__{joint_name}"] = float(sim_targets[index, joint_index])
                    record[f"replay_target__{joint_name}"] = float(replay_targets[index, joint_index])
                writer.writerow(record)
        print("Replay CSV:", output_path)

    if not passed:
        print("SIMULATION REPLAY COMPARISON FAILED")
        return 1
    print("SIMULATION REPLAY COMPARISON OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
