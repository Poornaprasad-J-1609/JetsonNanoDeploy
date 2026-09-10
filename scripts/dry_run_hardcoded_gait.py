#!/usr/bin/env python3
"""Hardware-free randomized audit of the hardcoded gait target pipeline."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardcoded_gait import HardcodedGaitPlayer  # noqa: E402
from joint_mapping import POLICY_JOINT_ORDER  # noqa: E402
from safety_monitor import SafetyMonitor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "hardcoded_gait_july13_minimal.yaml"),
    )
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=8675)
    parser.add_argument("--dt-jitter-ms", type=float, default=2.0)
    parser.add_argument("--amplitude-scale", type=float, default=None)
    parser.add_argument("--frequency-scale", type=float, default=None)
    parser.add_argument("--json", dest="json_path")
    return parser.parse_args()


def load_policy_gains():
    cfg = yaml.safe_load((ROOT / "config" / "mit_motor_control.yaml").read_text())
    policy = cfg["gains"]["policy"]
    kp, kd = [], []
    for name in POLICY_JOINT_ORDER:
        group = "hip" if "_hip_" in name else "thigh" if "_thigh_" in name else "calf"
        values = dict(policy[group])
        values.update((policy.get("joints", {}) or {}).get(name, {}))
        kp.append(float(values["kp"]))
        kd.append(float(values["kd"]))
    support = cfg.get("pose_support", {}).get("stand_joint_tau_ff", {}) or {}
    tau_support = [float(support.get(name, 0.0)) for name in POLICY_JOINT_ORDER]
    return np.asarray(kp), np.asarray(kd), np.asarray(tau_support)


def run_audit(args):
    if args.runs < 1 or args.seconds <= 0.0 or args.dt_jitter_ms < 0.0:
        raise ValueError("runs and seconds must be positive; jitter must be non-negative")
    rng = np.random.default_rng(args.seed)
    safety = SafetyMonitor(POLICY_JOINT_ORDER, control_dt=0.02)
    kp, kd, tau_support = load_policy_gains()
    total_samples = 0
    position_clips = 0
    rate_clips = 0
    max_step = 0.0
    max_speed = np.zeros(12)
    max_acceleration = np.zeros(12)
    max_one_cycle_torque = np.zeros(12)
    q_seen_min = np.full(12, np.inf)
    q_seen_max = np.full(12, -np.inf)

    for run in range(args.runs):
        player = HardcodedGaitPlayer.from_yaml(
            args.config,
            amplitude_scale=args.amplitude_scale,
            frequency_scale=args.frequency_scale,
        )
        previous = np.zeros(12, dtype=np.float32)
        previous_velocity = np.zeros(12, dtype=np.float32)
        elapsed = 0.0
        switch_time = rng.uniform(args.seconds * 0.3, args.seconds * 0.7)
        initial_direction = 1.0 if run % 2 == 0 else -1.0
        while elapsed < args.seconds:
            dt = float(np.clip(rng.normal(0.02, args.dt_jitter_ms / 1000.0), 0.005, 0.04))
            direction = initial_direction if elapsed < switch_time else -initial_direction
            raw = player.update(direction, dt, current_target=previous)
            clipped = safety.clip_q_target(raw)
            safe = safety.rate_limit_q_target(clipped, previous)
            position_clips += int(np.any(np.abs(clipped - raw) > 1.0e-7))
            rate_clips += int(np.any(np.abs(safe - clipped) > 1.0e-7))
            velocity = (safe - previous) / dt
            acceleration = (velocity - previous_velocity) / dt
            # Diagnostic only: one-cycle-lag PD estimate, not measured torque.
            tau = kp * (safe - previous) - kd * previous_velocity + tau_support
            max_one_cycle_torque = np.maximum(max_one_cycle_torque, np.abs(tau))
            max_step = max(max_step, float(np.max(np.abs(safe - previous))))
            max_speed = np.maximum(max_speed, np.abs(velocity))
            max_acceleration = np.maximum(max_acceleration, np.abs(acceleration))
            q_seen_min = np.minimum(q_seen_min, safe)
            q_seen_max = np.maximum(q_seen_max, safe)
            if not np.all(np.isfinite(safe)):
                raise RuntimeError(f"run {run}: non-finite target")
            previous = safe
            previous_velocity = velocity
            elapsed += dt
            total_samples += 1

    passed = position_clips == 0 and rate_clips == 0
    report = {
        "passed": passed,
        "runs": args.runs,
        "samples": total_samples,
        "seed": args.seed,
        "dt_jitter_ms": args.dt_jitter_ms,
        "position_limit_interventions": position_clips,
        "rate_limit_interventions": rate_clips,
        "maximum_target_step_rad": max_step,
        "joint_metrics": {
            name: {
                "target_min_rad": float(q_seen_min[i]),
                "target_max_rad": float(q_seen_max[i]),
                "maximum_speed_rad_s": float(max_speed[i]),
                "maximum_acceleration_rad_s2": float(max_acceleration[i]),
                "one_cycle_lag_torque_estimate_nm": float(max_one_cycle_torque[i]),
            }
            for i, name in enumerate(POLICY_JOINT_ORDER)
        },
        "note": "Torque values are dry-run one-cycle-lag estimates, not measured motor torque.",
    }
    return report


def main():
    args = parse_args()
    report = run_audit(args)
    print("HARDCODED GAIT DRY-RUN AUDIT")
    print(f"runs: {report['runs']}")
    print(f"samples: {report['samples']}")
    print(f"position_limit_interventions: {report['position_limit_interventions']}")
    print(f"rate_limit_interventions: {report['rate_limit_interventions']}")
    print(f"maximum_target_step_rad: {report['maximum_target_step_rad']:.6f}")
    print("joint                 target range rad       max vel   max accel  lag-tau est")
    for name, values in report["joint_metrics"].items():
        print(
            f"{name:20s} {values['target_min_rad']:+.4f}..{values['target_max_rad']:+.4f} "
            f"{values['maximum_speed_rad_s']:8.3f} {values['maximum_acceleration_rad_s2']:10.2f} "
            f"{values['one_cycle_lag_torque_estimate_nm']:11.2f}"
        )
    print("RESULT:", "PASS" if report["passed"] else "FAIL")
    print(report["note"])
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("JSON:", path)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
