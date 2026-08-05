#!/usr/bin/env python3
"""Replay real Grallator sensor logs through the current deployment pipeline."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from can_topology import resolve_joint_can_bus  # noqa: E402
from main_controller import (  # noqa: E402
    clip_policy_hip_actions,
    filtered_policy_action,
    shifted_safety_filter_with_diagnostics,
)
from motor_command_layer import MotorCommandLayer  # noqa: E402
from policy_runner import PolicyRunner  # noqa: E402
from safety_monitor import SafetyMonitor  # noqa: E402


COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "vx+0.05": (0.05, 0.0, 0.0),
    "vx+0.10": (0.10, 0.0, 0.0),
    "vx+0.20": (0.20, 0.0, 0.0),
    "vx-0.10": (-0.10, 0.0, 0.0),
    "vy+0.10": (0.0, 0.10, 0.0),
    "vy-0.10": (0.0, -0.10, 0.0),
}


def number(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def load_rows(paths):
    loaded = []
    for path in paths:
        with Path(path).expanduser().open(newline="", encoding="utf-8") as stream:
            loaded.extend(csv.DictReader(stream))
    return loaded


def policy_sensor_frames(rows):
    frames = []
    for row in rows:
        values = np.asarray(
            [number(row, f"obs_{index:03d}") for index in range(3, 36)],
            dtype=np.float32,
        )
        if row.get("mode") == "policy" and np.all(np.isfinite(values)):
            frames.append(values)
    if not frames:
        raise ValueError("logs contain no complete real policy sensor rows")
    return np.asarray(frames, dtype=np.float32)


def feedback_snapshot(order, q, qd):
    return {
        name: {
            "position_raw": float(q[index]),
            "joint_position": float(q[index]),
            "joint_velocity": float(qd[index]),
        }
        for index, name in enumerate(order)
    }


def create_runtime():
    runner = PolicyRunner()
    safety = SafetyMonitor(runner.policy_order, control_dt=runner.control_dt)
    motor_ids = yaml.safe_load((ROOT / "config" / "motor_ids.yaml").read_text())[
        "motor_ids"
    ]
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=list(runner.policy_order),
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    return runner, safety, layer


def replay_policy(args, rows, runner, safety, layer):
    frames = policy_sensor_frames(rows)
    rng = np.random.default_rng(args.seed)
    reports = []

    for label, command_values in COMMANDS.items():
        counts = {
            "joint_samples": 0,
            "actor_clip": 0,
            "hip_conditioned": 0,
            "action_filtered": 0,
            "joint_limited": 0,
            "rate_limited": 0,
            "torque_limited": 0,
        }
        max_tau = 0.0
        min_impedance = 1.0
        max_actor = 0.0
        max_target = 0.0
        command = np.asarray(command_values, dtype=np.float32)

        for _ in range(args.trials):
            previous_raw = np.zeros(12, dtype=np.float32)
            previous_sent = np.zeros(12, dtype=np.float32)
            previous_target = runner.q_stand.copy()
            held_feedback = None
            gyro_bias = rng.uniform(-0.03, 0.03, 3)
            q_bias = rng.uniform(-0.01, 0.01, 12)
            qd_bias = rng.uniform(-0.05, 0.05, 12)

            for frame in frames[:: args.row_stride]:
                gyro = frame[0:3] + gyro_bias + rng.normal(0.0, 0.02, 3)
                gravity = frame[3:6] + rng.normal(0.0, 0.0087, 3)
                gravity /= max(float(np.linalg.norm(gravity)), 1.0e-6)
                q = frame[9:21] + q_bias + rng.normal(0.0, 0.005, 12)
                qd = frame[21:33] + qd_bias + rng.normal(0.0, 0.12, 12)
                if held_feedback is not None and rng.random() < 0.01:
                    q, qd = held_feedback
                else:
                    held_feedback = (q.copy(), qd.copy())

                observation = runner.build_observation(
                    base_ang_vel_b=gyro,
                    projected_gravity_b=gravity,
                    command=command,
                    q_current=q,
                    qd_current=qd,
                    previous_action=previous_raw,
                )
                raw_action = runner.infer_action(observation)
                control_action = clip_policy_hip_actions(
                    raw_action,
                    runner.policy_order,
                    hip_clip_abs=args.hip_clip,
                    hip_scale=args.hip_scale,
                )
                sent_action = filtered_policy_action(
                    raw_action=control_action,
                    previous_action=previous_sent,
                    clip_abs=args.action_clip,
                    smoothing=args.smoothing,
                    delta_limit_abs=args.delta_limit,
                )
                requested = runner.action_to_q_target(sent_action)
                safe, diagnostics = shifted_safety_filter_with_diagnostics(
                    safety,
                    requested,
                    previous_target,
                    np.zeros(12, dtype=np.float32),
                    apply_rate_limit=True,
                    use_policy_limits=False,
                )
                commands = layer.build_mit_commands(
                    safe,
                    phase="policy",
                    feedback_by_joint=feedback_snapshot(runner.policy_order, q, qd),
                    previous_command_q=previous_target,
                    max_command_delta=safety.dq_max,
                )

                counts["joint_samples"] += 12
                counts["actor_clip"] += int(np.sum(np.abs(raw_action) > args.action_clip))
                counts["hip_conditioned"] += int(
                    np.sum(np.abs(control_action - raw_action) > 1.0e-6)
                )
                counts["action_filtered"] += int(
                    np.sum(np.abs(sent_action - control_action) > 1.0e-6)
                )
                counts["joint_limited"] += int(
                    np.sum(diagnostics["target_joint_limited"])
                )
                counts["rate_limited"] += int(
                    np.sum(diagnostics["target_rate_limited"])
                )
                counts["torque_limited"] += sum(
                    bool(item["torque_limited"]) for item in commands
                )
                max_tau = max(
                    max_tau,
                    max(abs(float(item["tau_pd_est"])) for item in commands),
                )
                min_impedance = min(
                    min_impedance,
                    min(float(item["impedance_scale"]) for item in commands),
                )
                max_actor = max(max_actor, float(np.max(np.abs(raw_action))))
                max_target = max(max_target, float(np.max(np.abs(requested))))
                previous_raw = raw_action
                previous_sent = sent_action
                previous_target = np.asarray(
                    [item["q_des"] for item in commands], dtype=np.float32
                )

        denominator = max(1, counts["joint_samples"])
        reports.append(
            {
                "command": label,
                **{
                    key + "_pct": 100.0 * counts[key] / denominator
                    for key in counts
                    if key != "joint_samples"
                },
                "max_estimated_torque_nm": max_tau,
                "minimum_impedance_scale": min_impedance,
                "max_raw_action": max_actor,
                "max_requested_target_rad": max_target,
            }
        )
    return reports


def replay_pose(rows, runner, layer):
    support_cfg = layer.cfg.get("pose_support", {}) or {}
    support_map = support_cfg.get("stand_joint_tau_ff", {}) or {}
    support = np.asarray(
        [float(support_map.get(name, 0.0)) for name in runner.policy_order],
        dtype=np.float32,
    )
    counts = {"samples": 0, "torque_limited": 0}
    max_tau = 0.0
    min_impedance = 1.0
    for row in rows:
        phase = row.get("mode")
        if phase not in ("sit", "stand"):
            continue
        q = np.asarray([number(row, f"q_{i:02d}") for i in range(12)])
        qd = np.asarray([number(row, f"qd_{i:02d}") for i in range(12)])
        target = np.asarray([number(row, f"q_target_{i:02d}") for i in range(12)])
        if not np.all(np.isfinite(np.concatenate((q, qd, target)))):
            continue
        for support_scale in np.linspace(0.0, 1.0, 5):
            commands = layer.build_mit_commands(
                target,
                phase=phase,
                feedback_by_joint=feedback_snapshot(runner.policy_order, q, qd),
                joint_feedforward_torque_target=support_scale * support,
            )
            counts["samples"] += len(commands)
            counts["torque_limited"] += sum(
                bool(item["torque_limited"]) for item in commands
            )
            max_tau = max(
                max_tau,
                max(abs(float(item["tau_pd_est"])) for item in commands),
            )
            min_impedance = min(
                min_impedance,
                min(float(item["impedance_scale"]) for item in commands),
            )
    return {
        "joint_samples": counts["samples"],
        "torque_limited_pct": (
            100.0 * counts["torque_limited"] / max(1, counts["samples"])
        ),
        "max_estimated_torque_nm": max_tau,
        "minimum_impedance_scale": min_impedance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--row-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11078)
    parser.add_argument("--action-clip", type=float, default=3.2)
    parser.add_argument("--hip-clip", type=float, default=1.6)
    parser.add_argument("--hip-scale", type=float, default=0.30)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--delta-limit", type=float, default=0.20)
    args = parser.parse_args()
    if args.trials < 1 or args.row_stride < 1:
        parser.error("--trials and --row-stride must be positive")

    rows = load_rows(args.logs)
    runner, safety, layer = create_runtime()
    reports = replay_policy(args, rows, runner, safety, layer)
    pose_report = replay_pose(rows, runner, layer)

    print(f"policy_sha256: {runner.policy_sha256}")
    print(f"real_log_rows: {len(rows)}")
    print(f"monte_carlo_trials_per_command: {args.trials}")
    for report in reports:
        print(
            "{command:9s} torque_limit={torque_limited_pct:6.3f}% "
            "min_stiffness={minimum_impedance_scale:6.3f} "
            "joint_limit={joint_limited_pct:6.3f}% "
            "rate_limit={rate_limited_pct:6.3f}% "
            "hip_conditioning={hip_conditioned_pct:6.3f}% "
            "action_filter={action_filtered_pct:6.3f}% "
            "max_tau={max_estimated_torque_nm:7.2f}Nm "
            "max_action={max_raw_action:6.3f}".format(**report)
        )
    print(
        "pose      torque_limit={torque_limited_pct:6.3f}% "
        "min_stiffness={minimum_impedance_scale:6.3f} "
        "max_tau={max_estimated_torque_nm:7.2f}Nm samples={joint_samples}".format(
            **pose_report
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
