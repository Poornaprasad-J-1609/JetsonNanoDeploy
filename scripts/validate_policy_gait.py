#!/usr/bin/env python3
"""No-hardware deployment gait replay for the verified 48D Grallator policy."""

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
from main_controller import action_equivalent_for_q_target, shifted_safety_filter, smoothstep
from motor_command_layer import MotorCommandLayer
from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def feedback_snapshot(policy_order, q, qd):
    now = time.monotonic()
    feedback = {}
    for index, joint_name in enumerate(policy_order):
        feedback[joint_name] = {
            "timestamp": now,
            "fault_bits": 0,
            "joint_position": float(q[index]),
            "joint_velocity": float(qd[index]),
            "position": float(q[index]),
            "velocity": float(qd[index]),
            "torque": 0.0,
            "position_raw": float(q[index]),
            "velocity_raw": float(qd[index]),
            "torque_raw": 0.0,
        }
    return feedback


def phase_correlation(series_a, series_b):
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    if a.size < 3 or b.size < 3:
        return np.nan
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, default=0.35)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--entry-seconds", type=float, default=2.0)
    parser.add_argument("--stand-settle-seconds", type=float, default=0.5)
    parser.add_argument(
        "--plant-response",
        type=float,
        default=0.10,
        help=(
            "fraction of the final target followed by fake joint feedback each "
            "50 Hz cycle; 1.0 is an unrealistic teleporting plant"
        ),
    )
    parser.add_argument("--csv", default=str(ROOT / "logs" / "policy_gait_replay.csv"))
    parser.add_argument("--allow-policy-hash-mismatch", action="store_true")
    args = parser.parse_args()

    runner = PolicyRunner(allow_policy_hash_mismatch=args.allow_policy_hash_mismatch)
    if runner.observation_dim != 48 or runner.action_dim != 12:
        print("FAIL: policy contract is not 48 obs / 12 action")
        return 2
    if not np.isclose(runner.control_dt, 0.02):
        print(f"FAIL: policy control_dt is {runner.control_dt}, expected 0.02")
        return 2

    four_bar_cfg = load_yaml(ROOT / "config" / "four_bar_transmission.yaml")
    if bool(four_bar_cfg["four_bar_transmission"].get("enabled", False)):
        print("FAIL: four-bar transmission is enabled; replay expects 1:1 path")
        return 2

    motor_cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    motor_layer = MotorCommandLayer(
        runner.policy_order,
        motor_cfg["motor_ids"],
        active_joints=[],
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )
    safety = SafetyMonitor(runner.policy_order, control_dt=runner.control_dt)

    dt = float(runner.control_dt)
    total_steps = int(round(float(args.duration) / dt))
    stand_steps = int(round(float(args.stand_settle_seconds) / dt))
    entry_steps = max(1, int(round(float(args.entry_seconds) / dt)))

    q = runner.q_stand.astype(np.float32).copy()
    qd = np.zeros(12, dtype=np.float32)
    previous_raw_action = np.zeros(12, dtype=np.float32)
    q_entry = q.copy()
    q_previous_target = q.copy()
    plant_response = float(np.clip(args.plant_response, 0.01, 1.0))

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    q_history = []
    action_history = []
    sent_action_history = []
    policy_clip_counts = np.zeros(12, dtype=np.int64)
    torque_clip_counts = np.zeros(12, dtype=np.int64)
    steady_steps = 0
    failures = []

    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    for step in range(total_steps):
        walking = step >= stand_steps
        entry_elapsed_steps = max(0, step - stand_steps + 1)
        alpha = 0.0 if not walking else smoothstep(min(1.0, entry_elapsed_steps / entry_steps))
        policy_command = (
            runner.policy_command
            if runner.autonomous_march
            else command
            if walking
            else np.zeros(3, dtype=np.float32)
        )

        obs = runner.build_observation(
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            command=policy_command,
            q_current=q,
            qd_current=qd,
            previous_action=previous_raw_action,
            marching_clock=runner.marching_clock(
                max(0, step - stand_steps) * runner.control_dt,
                march_enabled=walking,
            ),
        )
        raw_action = runner.infer_action(obs)
        if not np.all(np.isfinite(raw_action)):
            failures.append(f"step {step}: non-finite raw action")
            break

        q_actor = runner.action_to_q_target(raw_action)
        if walking:
            q_requested = ((1.0 - alpha) * q_entry + alpha * q_actor).astype(np.float32)
            sent_action = (
                raw_action.copy()
                if alpha >= 0.999
                else action_equivalent_for_q_target(runner, q_requested)
            )
        else:
            q_requested = q_previous_target.copy()
            sent_action = np.zeros(12, dtype=np.float32)

        q_safety = shifted_safety_filter(
            safety,
            q_requested,
            q_previous_target,
            np.zeros(12, dtype=np.float32),
            apply_rate_limit=bool(alpha < 0.999),
            use_policy_limits=True,
        )
        commands = motor_layer.build_mit_commands(
            q_safety,
            phase="policy" if walking else "hold",
            feedback_by_joint=feedback_snapshot(runner.policy_order, q, qd),
        )

        q_sent = q_safety.copy()
        for item in commands:
            index = motor_layer.policy_index_by_joint[item["joint_name"]]
            q_sent[index] = float(item["q_des"])

        if walking and alpha >= 0.999:
            steady_steps += 1
            if not np.allclose(sent_action, raw_action, atol=1e-6):
                failures.append(f"step {step}: sent action differs from raw action after entry")
            policy_clip_counts += (np.abs(q_safety - q_requested) > 1e-5)
            for item in commands:
                if item.get("torque_limited"):
                    index = motor_layer.policy_index_by_joint[item["joint_name"]]
                    torque_clip_counts[index] += 1

        q_next = (q + plant_response * (q_sent - q)).astype(np.float32)
        qd = ((q_next - q) / dt).astype(np.float32)
        q = q_next.copy()
        q_previous_target = q_sent.astype(np.float32).copy()
        previous_raw_action = raw_action.astype(np.float32).copy()

        q_history.append(q.copy())
        action_history.append(raw_action.copy())
        sent_action_history.append(sent_action.copy())
        row = {
            "step": step,
            "time_s": step * dt,
            "walking": int(walking),
            "entry_alpha": alpha,
            "command_vx": float(policy_command[0]),
            "command_vy": float(policy_command[1]),
            "command_yaw": float(policy_command[2]),
        }
        for index in range(48):
            row[f"obs_{index:03d}"] = float(obs[index])
        for index, joint_name in enumerate(runner.policy_order):
            row[f"raw_action_{index:02d}"] = float(raw_action[index])
            row[f"sent_action_{index:02d}"] = float(sent_action[index])
            row[f"q_actor_{index:02d}"] = float(q_actor[index])
            row[f"q_requested_{index:02d}"] = float(q_requested[index])
            row[f"q_safety_{index:02d}"] = float(q_safety[index])
            row[f"q_sent_{index:02d}"] = float(q_sent[index])
            row[f"{joint_name}_q_sent"] = float(q_sent[index])
        rows.append(row)

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    q_history = np.asarray(q_history, dtype=np.float32)
    action_history = np.asarray(action_history, dtype=np.float32)
    sent_action_history = np.asarray(sent_action_history, dtype=np.float32)
    steady_start = min(len(q_history), stand_steps + entry_steps)
    steady_q = q_history[steady_start:]
    steady_action = action_history[steady_start:]
    steady_sent = sent_action_history[steady_start:]

    if steady_q.shape[0] < int(round(5.0 / dt)):
        failures.append("less than 5 seconds of steady-state walking after entry")
    if steady_q.size and float(np.ptp(steady_q, axis=0).max()) < 1e-3:
        failures.append("joint targets are effectively constant")
    if steady_action.size and steady_sent.size:
        max_action_error = float(np.max(np.abs(steady_action - steady_sent)))
        if max_action_error > 1e-5:
            failures.append(f"raw/sent action mismatch after entry: {max_action_error:.6g}")
    else:
        max_action_error = np.nan

    leg_indices = {
        leg: [i for i, name in enumerate(runner.policy_order) if name.startswith(leg + "_")]
        for leg in ("FR", "FL", "BR", "BL")
    }
    p2p_by_leg = {}
    for leg, indices in leg_indices.items():
        if steady_q.size:
            p2p_by_leg[leg] = float(np.ptp(steady_q[:, indices], axis=0).max())
            if p2p_by_leg[leg] < 1e-3:
                failures.append(f"{leg} commands are effectively constant")
        else:
            p2p_by_leg[leg] = np.nan

    clip_pct = 100.0 * policy_clip_counts / max(1, steady_steps)
    torque_clip_pct = 100.0 * torque_clip_counts / max(1, steady_steps)
    if steady_steps > 0 and np.any(clip_pct > 1.0):
        clipped = [
            f"{runner.policy_order[i]}={clip_pct[i]:.2f}%"
            for i in np.where(clip_pct > 1.0)[0]
        ]
        failures.append("policy target clipped above 1% steady-state: " + ", ".join(clipped))

    calf_series = {
        leg: steady_q[:, indices[-1]] if steady_q.size and indices else []
        for leg, indices in leg_indices.items()
    }
    correlations = {
        "FR_FL": phase_correlation(calf_series["FR"], calf_series["FL"]),
        "FR_BL": phase_correlation(calf_series["FR"], calf_series["BL"]),
        "FR_BR": phase_correlation(calf_series["FR"], calf_series["BR"]),
        "FL_BR": phase_correlation(calf_series["FL"], calf_series["BR"]),
    }

    print("Policy gait replay")
    print("  policy:", runner.policy_path)
    print("  sha256:", runner.policy_sha256)
    print("  command:", [float(x) for x in command])
    print("  steps:", total_steps, "steady_steps:", steady_steps)
    print("  fake plant response:", f"{plant_response:.3f} per cycle")
    print("  csv:", csv_path)
    print("  max raw/sent action error after entry:", f"{max_action_error:.6g}")
    print("  peak-to-peak joint target by leg:")
    for leg in ("FR", "FL", "BR", "BL"):
        print(f"    {leg}: {p2p_by_leg[leg]:.4f} rad")
    print("  calf target phase correlations:")
    for name, value in correlations.items():
        print(f"    {name}: {value:+.3f}")
    print("  policy target clip percent by joint:")
    for joint_name, pct in zip(runner.policy_order, clip_pct):
        print(f"    {joint_name}: {pct:.2f}%")
    print("  torque limiter percent by joint:")
    for joint_name, pct in zip(runner.policy_order, torque_clip_pct):
        print(f"    {joint_name}: {pct:.2f}%")

    if failures:
        print("FAIL:")
        for failure in failures:
            print("  -", failure)
        return 1
    print("PASS: no-hardware policy gait replay contract satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
