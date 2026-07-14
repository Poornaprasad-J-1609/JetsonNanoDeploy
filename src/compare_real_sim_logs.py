#!/usr/bin/env python3
"""Compare IsaacLab policy/motor logs with one deployment CSV log.

The comparison is intentionally observation-first:

* sim policy rows use obs[9:12] as the authoritative actor command;
* real deployment rows use obs[9:12] as the authoritative actor command;
* obs[3:6] and obs[6:9] are compared as the actual IMU values sent to policy.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def read_rows(path):
    with Path(path).open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def value(row, key, default=np.nan):
    raw = row.get(key, "")
    if raw == "" or raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def vector(row, prefix, count, width):
    return np.asarray(
        [value(row, f"{prefix}_{index:0{width}d}") for index in range(count)],
        dtype=np.float32,
    )


def finite(values):
    if not isinstance(values, np.ndarray):
        values = list(values)
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def stats(values):
    values = finite(values)
    if values.size == 0:
        return "na"
    return (
        f"mean={values.mean():+.4f} "
        f"p50={np.percentile(values, 50):+.4f} "
        f"p95={np.percentile(values, 95):+.4f} "
        f"max={values.max():+.4f}"
    )


def command_masks(commands):
    magnitude = np.max(np.abs(commands), axis=1)
    return {
        "forward": commands[:, 0] > 0.02,
        "backward": commands[:, 0] < -0.02,
        "stop": magnitude <= 0.02,
    }


def print_policy_summary(title, commands, gyro, gravity, action, q, qd, extra=None):
    print(f"\n== {title} ==")
    for label, mask in command_masks(commands).items():
        if not np.any(mask):
            continue
        print(f"{label}: rows={int(np.count_nonzero(mask))}")
        print("  policy_cmd_vx:     ", stats(commands[mask, 0]))
        print("  policy_cmd_vy:     ", stats(commands[mask, 1]))
        print("  policy_cmd_yaw:    ", stats(commands[mask, 2]))
        print("  gyro_norm:         ", stats(np.linalg.norm(gyro[mask], axis=1)))
        print("  gravity_xy_norm:   ", stats(np.linalg.norm(gravity[mask, :2], axis=1)))
        print("  action_abs_max:    ", stats(np.max(np.abs(action[mask]), axis=1)))
        print("  q_abs_max:         ", stats(np.max(np.abs(q[mask]), axis=1)))
        print("  qd_abs_max:        ", stats(np.max(np.abs(qd[mask]), axis=1)))
        if extra:
            for name, series in extra.items():
                print(f"  {name:<18}", stats(series[mask]))


def summarize_sim_policy(rows, force_zero_base_lin_vel=False):
    obs = np.stack([vector(row, "obs", 48, 3) for row in rows])
    original_base_lin_vel_abs_max = float(np.max(np.abs(obs[:, 0:3])))
    if force_zero_base_lin_vel:
        obs[:, 0:3] = 0.0
    actions = np.stack([
        vector(row, "policy_action", 12, 2)
        if "policy_action_00" in row
        else vector(row, "action", 12, 2)
        for row in rows
    ])
    return {
        "commands": obs[:, 9:12],
        "gyro": obs[:, 3:6],
        "gravity": obs[:, 6:9],
        "actions": actions,
        "q": obs[:, 12:24],
        "qd": obs[:, 24:36],
        "original_base_lin_vel_abs_max": original_base_lin_vel_abs_max,
        "base_lin_vel_forced_zero": bool(force_zero_base_lin_vel),
    }


def summarize_real_policy(rows):
    rows = [row for row in rows if row.get("mode") == "policy"]
    if not rows:
        raise ValueError("real log has no rows with mode=policy")
    obs = np.stack([vector(row, "obs", 48, 3) for row in rows])
    return {
        "commands": obs[:, 9:12],
        "gyro": obs[:, 3:6],
        "gravity": obs[:, 6:9],
        "actions": np.stack([vector(row, "action", 12, 2) for row in rows]),
        "sent_actions": np.stack([vector(row, "sent_action", 12, 2) for row in rows]),
        "q": np.stack([vector(row, "q", 12, 2) for row in rows]),
        "qd": np.stack([vector(row, "qd", 12, 2) for row in rows]),
        "tau_cmd_max": np.asarray([value(row, "tau_cmd_max") for row in rows]),
        "tau_fb_max": np.asarray([value(row, "tau_fb_max") for row in rows]),
        "feedback_age_ms": np.asarray([value(row, "feedback_age_max_ms") for row in rows]),
        "loop_hz": np.asarray([value(row, "loop_hz") for row in rows]),
    }


def summarize_sim_motor(rows):
    print("\n== SIM motor torque/velocity by command metadata ==")
    command_vx = np.asarray([value(row, "command_vx") for row in rows])
    for label, mask in {
        "forward": command_vx > 0.02,
        "backward": command_vx < -0.02,
        "stop": np.abs(command_vx) <= 0.02,
    }.items():
        if not np.any(mask):
            continue
        masked = [row for row, keep in zip(rows, mask) if keep]
        print(f"{label}: motor_rows={len(masked)}")
        for field in (
            "applied_torque_postclip",
            "computed_torque_preclip",
            "qd",
            "q",
        ):
            print(f"  abs({field}):", stats(abs(value(row, field)) for row in masked))


def print_gap_hints(sim, real):
    print("\n== Gap hints ==")
    for label, sim_mask in command_masks(sim["commands"]).items():
        real_mask = command_masks(real["commands"]).get(label)
        if not np.any(sim_mask) or real_mask is None or not np.any(real_mask):
            continue
        sim_vx = np.nanmean(np.abs(sim["commands"][sim_mask, 0]))
        real_vx = np.nanmean(np.abs(real["commands"][real_mask, 0]))
        sim_grav = np.nanpercentile(np.linalg.norm(sim["gravity"][sim_mask, :2], axis=1), 95)
        real_grav = np.nanpercentile(np.linalg.norm(real["gravity"][real_mask, :2], axis=1), 95)
        sim_qd = np.nanpercentile(np.max(np.abs(sim["qd"][sim_mask]), axis=1), 95)
        real_qd = np.nanpercentile(np.max(np.abs(real["qd"][real_mask]), axis=1), 95)
        real_raw = np.nanpercentile(np.max(np.abs(real["actions"][real_mask]), axis=1), 95)
        real_sent = np.nanpercentile(np.max(np.abs(real["sent_actions"][real_mask]), axis=1), 95)
        print(
            f"{label}: |cmd_vx| real/sim={real_vx:.3f}/{sim_vx:.3f}, "
            f"gravity_xy_p95 real/sim={real_grav:.3f}/{sim_grav:.3f}, "
            f"qd_abs_p95 real/sim={real_qd:.3f}/{sim_qd:.3f}, "
            f"real sent/raw action p95={real_sent:.3f}/{real_raw:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_policy_log")
    parser.add_argument("sim_motor_log")
    parser.add_argument("real_log")
    parser.add_argument(
        "--force-zero-sim-base-lin-vel",
        action="store_true",
        help="force sim obs[0:3] to zero before summarizing policy inputs",
    )
    args = parser.parse_args()

    sim_policy = summarize_sim_policy(
        read_rows(args.sim_policy_log),
        force_zero_base_lin_vel=args.force_zero_sim_base_lin_vel,
    )
    real_policy = summarize_real_policy(read_rows(args.real_log))

    print(
        "SIM obs[0:3] original abs max:",
        f"{sim_policy['original_base_lin_vel_abs_max']:.6f}",
        "forced_zero=",
        sim_policy["base_lin_vel_forced_zero"],
    )
    print_policy_summary(
        "SIM policy observation/action",
        sim_policy["commands"],
        sim_policy["gyro"],
        sim_policy["gravity"],
        sim_policy["actions"],
        sim_policy["q"],
        sim_policy["qd"],
    )
    print_policy_summary(
        "REAL deployment observation/action",
        real_policy["commands"],
        real_policy["gyro"],
        real_policy["gravity"],
        real_policy["actions"],
        real_policy["q"],
        real_policy["qd"],
        extra={
            "sent_action_abs_max": np.max(np.abs(real_policy["sent_actions"]), axis=1),
            "tau_cmd_max": real_policy["tau_cmd_max"],
            "tau_fb_max": real_policy["tau_fb_max"],
            "feedback_age_ms": real_policy["feedback_age_ms"],
            "loop_hz": real_policy["loop_hz"],
        },
    )
    summarize_sim_motor(read_rows(args.sim_motor_log))
    print_gap_hints(sim_policy, real_policy)


if __name__ == "__main__":
    raise SystemExit(main())
