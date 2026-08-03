#!/usr/bin/env python3
"""Audit the exact 48-value observation recorded by the hardware controller."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from joint_mapping import POLICY_JOINT_ORDER  # noqa: E402


FIELDS = (
    (0, 3, "base_lin_vel", "m/s; required zero"),
    (3, 6, "base_ang_vel", "rad/s"),
    (6, 9, "projected_gravity", "unit vector"),
    (9, 12, "command", "m/s,m/s,rad/s"),
    (12, 24, "joint_pos_rel", "rad"),
    (24, 36, "joint_vel", "rad/s"),
    (36, 48, "previous_applied_action", "actor-coordinate action"),
)


def _number(row, key, default=np.nan):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def load_policy_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("obs_000", "") != ""]
    if not rows:
        raise ValueError("CSV contains no rows with obs_000..obs_047")
    return rows


def _matrix(rows, prefix, count, width):
    return np.asarray(
        [[_number(row, f"{prefix}_{index:0{width}d}") for index in range(count)] for row in rows],
        dtype=np.float64,
    )


def _correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3:
        return np.nan
    a = a[valid]
    b = b[valid]
    if np.std(a) < 1.0e-8 or np.std(b) < 1.0e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def velocity_validation(q, qd, timestamps, steps=None):
    sparse_logging = False
    if steps is not None and len(steps) > 1:
        step_delta = np.diff(np.asarray(steps, dtype=np.float64))
        valid_step_delta = step_delta[np.isfinite(step_delta) & (step_delta > 0.0)]
        sparse_logging = bool(
            valid_step_delta.size and np.nanmedian(valid_step_delta) > 1.5
        )
    # At full-rate logging, finite-difference position and reported velocity
    # describe adjacent 20 ms cycles. Sparse logs compare an interval-average
    # position velocity with one instantaneous sample, so allow the small sign
    # disagreement observed around turning points while retaining correlation
    # and gain checks that still catch an inverted encoder direction.
    minimum_sign_agreement = 0.90 if sparse_logging else 0.95
    result = []
    for index, joint_name in enumerate(POLICY_JOINT_ORDER):
        dt = np.diff(timestamps)
        dq = np.diff(q[:, index])
        valid = np.isfinite(dt) & np.isfinite(dq) & (dt > 1.0e-4) & (dt < 0.25)
        fd = np.full_like(dt, np.nan)
        fd[valid] = dq[valid] / dt[valid]
        reported = qd[1:, index]
        active = valid & np.isfinite(reported) & (np.abs(fd) > 0.02)
        sign_agreement = (
            float(np.mean(np.sign(fd[active]) == np.sign(reported[active])))
            if np.count_nonzero(active) >= 3 else np.nan
        )
        corr = _correlation(fd[valid], reported[valid])
        denom = float(np.sqrt(np.nanmean(fd[valid] ** 2))) if np.any(valid) else np.nan
        gain = (
            float(np.sqrt(np.nanmean(reported[valid] ** 2)) / denom)
            if np.isfinite(denom) and denom > 1.0e-6 else np.nan
        )
        rms = (
            float(np.sqrt(np.nanmean((reported[valid] - fd[valid]) ** 2)))
            if np.any(valid) else np.nan
        )
        passed = bool(
            np.count_nonzero(active) < 3
            or (
                np.isfinite(sign_agreement)
                and sign_agreement >= minimum_sign_agreement
                and (not np.isfinite(corr) or corr >= 0.70)
                and (not np.isfinite(gain) or 0.5 <= gain <= 2.0)
            )
        )
        result.append({
            "joint": joint_name,
            "correlation": corr,
            "gain_ratio": gain,
            "sign_agreement": sign_agreement,
            "rms_difference": rms,
            "time_delay_s": 0.0,
            "minimum_sign_agreement": minimum_sign_agreement,
            "pass": passed,
        })
    return result


def audit_rows(rows):
    obs = _matrix(rows, "obs", 48, 3)
    actions = _matrix(rows, "action", 12, 2)
    q = _matrix(rows, "q", 12, 2)
    qd = _matrix(rows, "qd", 12, 2)
    timestamps = np.asarray([_number(row, "elapsed_s", i * 0.02) for i, row in enumerate(rows)])
    steps = np.asarray([_number(row, "step", np.nan) for row in rows])
    warnings = []

    if obs.shape[1] != 48:
        raise ValueError(f"observation width is {obs.shape[1]}, expected 48")
    if not np.all(np.isfinite(obs)):
        warnings.append("observation contains NaN or Inf")

    configured_orders = {
        tuple(name for name in row.get("policy_joint_order", "").split(",") if name)
        for row in rows
    }
    configured_orders.discard(tuple())
    order_ok = not configured_orders or configured_orders == {POLICY_JOINT_ORDER}
    if not order_ok:
        warnings.append(f"policy joint order mismatch: {sorted(configured_orders)}")

    with (ROOT / "config" / "default_pose.yaml").open("r", encoding="utf-8") as stream:
        default_cfg = yaml.safe_load(stream)["default_pose"]
    q_default = np.asarray([default_cfg[name] for name in POLICY_JOINT_ORDER], dtype=np.float64)
    expected_q_rel = q - q_default
    q_rel_error = np.nanmax(np.abs(obs[:, 12:24] - expected_q_rel), axis=0)
    if np.nanmax(q_rel_error) > 2.0e-4:
        warnings.append("joint position observation does not match policy feedback minus training default")

    command = np.asarray(
        [[_number(row, "policy_vx"), _number(row, "policy_vy"), _number(row, "policy_yaw")] for row in rows]
    )
    command_error = float(np.nanmax(np.abs(obs[:, 9:12] - command)))
    if command_error > 2.0e-5:
        warnings.append("command field differs from displayed policy command")

    previous_error = 0.0
    previous_checked = False
    if len(rows) > 1:
        expected_previous = actions[:-1]
        adjacent = np.isfinite(steps[1:]) & np.isfinite(steps[:-1]) & (
            np.abs((steps[1:] - steps[:-1]) - 1.0) <= 1.0e-9
        )
        finite = np.all(np.isfinite(expected_previous), axis=1)
        comparable = adjacent & finite
        if np.any(comparable):
            previous_checked = True
            previous_error = float(
                np.nanmax(
                    np.abs(
                        obs[1:, 36:48][comparable]
                        - expected_previous[comparable]
                    )
                )
            )
            if previous_error > 2.0e-5:
                warnings.append(
                    "previous action does not match the previous raw actor output"
                )

    gravity_norm = np.linalg.norm(obs[:, 6:9], axis=1)
    if np.nanmax(np.abs(obs[:, 0:3])) > 1.0e-7:
        warnings.append("base linear velocity slots are not exactly zero")
    if np.nanmax(np.abs(gravity_norm - 1.0)) > 0.08:
        warnings.append("projected gravity norm is not near one")
    if np.nanmedian(obs[:, 8]) > -0.5:
        warnings.append("gravity sign is inconsistent with an upright robot")
    if np.nanmax(np.abs(obs[:, 24:36])) > 30.0:
        warnings.append("joint velocity magnitude is implausibly high for rad/s")

    movement = np.nanmax(q, axis=0) - np.nanmin(q, axis=0)
    qd_range = np.nanmax(qd, axis=0) - np.nanmin(qd, axis=0)
    for index, joint_name in enumerate(POLICY_JOINT_ORDER):
        if movement[index] > 0.03 and qd_range[index] < 1.0e-5:
            warnings.append(f"{joint_name}: joint velocity remains zero while position moves")

    velocity = velocity_validation(q, obs[:, 24:36], timestamps, steps=steps)
    for item in velocity:
        if not item["pass"]:
            warnings.append(
                f"{item['joint']}: velocity sign/scale validation failed "
                f"(corr={item['correlation']:.3f}, gain={item['gain_ratio']:.3f}, "
                f"sign={item['sign_agreement']:.1%})"
            )

    table = []
    joint_fields = {"joint_pos_rel", "joint_vel", "previous_applied_action"}
    for start, end, semantic, unit in FIELDS:
        for obs_index in range(start, end):
            joint_name = ""
            if semantic in joint_fields:
                joint_name = POLICY_JOINT_ORDER[obs_index - start]
            values = obs[:, obs_index]
            passed = bool(np.all(np.isfinite(values)))
            if semantic == "base_lin_vel":
                passed = passed and bool(np.nanmax(np.abs(obs[:, 0:3])) <= 1.0e-7)
            elif semantic == "projected_gravity":
                passed = passed and bool(np.nanmax(np.abs(gravity_norm - 1.0)) <= 0.08)
            elif semantic == "command":
                passed = passed and command_error <= 2.0e-5
            elif semantic == "joint_pos_rel":
                passed = passed and q_rel_error[obs_index - start] <= 2.0e-4 and order_ok
            elif semantic == "joint_vel":
                passed = passed and velocity[obs_index - start]["pass"] and order_ok
            elif semantic == "previous_applied_action":
                passed = passed and (
                    not previous_checked or previous_error <= 2.0e-5
                ) and order_ok
            table.append({
                "observation_index": obs_index,
                "semantic_field": semantic,
                "joint_name": joint_name,
                "minimum": float(np.nanmin(values)),
                "maximum": float(np.nanmax(values)),
                "mean": float(np.nanmean(values)),
                "standard_deviation": float(np.nanstd(values)),
                "expected_unit": unit,
                "result": "PASS" if passed else "FAIL",
            })
    return table, warnings, velocity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = load_policy_rows(args.csv)
    table, warnings, velocity = audit_rows(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "observation_audit.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    header = (
        "index field               joint               minimum    maximum       mean        std unit             result"
    )
    lines = [header, "-" * len(header)]
    for item in table:
        lines.append(
            f"{item['observation_index']:>5d} {item['semantic_field']:<19s} "
            f"{item['joint_name']:<19s} {item['minimum']:+10.4f} "
            f"{item['maximum']:+10.4f} {item['mean']:+10.4f} "
            f"{item['standard_deviation']:+10.4f} {item['expected_unit']:<16s} "
            f"{item['result']}"
        )
    lines.append("\nJOINT VELOCITY VALIDATION")
    for item in velocity:
        lines.append(
            f"{item['joint']}: corr={item['correlation']:+.4f} "
            f"gain={item['gain_ratio']:+.4f} sign={item['sign_agreement']:+.4f} "
            f"rms={item['rms_difference']:+.4f} delay={item['time_delay_s']:.4f}s "
            f"{'PASS' if item['pass'] else 'FAIL'}"
        )
    lines.append("\nWARNINGS")
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- none")
    report = "\n".join(lines) + "\n"
    print(report, end="")
    (output_dir / "observation_audit_report.txt").write_text(report, encoding="utf-8")
    print(f"Saved: {table_path}")
    return 1 if any(item["result"] == "FAIL" for item in table) else 0


if __name__ == "__main__":
    raise SystemExit(main())
