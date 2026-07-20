#!/usr/bin/env python3
"""Compare two completed suspended Grallator policy runs."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_runner import EXPECTED_POLICY_JOINT_ORDER  # noqa: E402


CONFIG_FIELDS = [
    "policy_sha256",
    "runtime_control_hz",
    "policy_action_scale",
    "exact_policy_after_entry",
    "can_topology",
    "can_backend",
    "imu",
]


def load_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise SystemExit(f"ERROR: {path} has no data rows")
    return rows, fieldnames


def first_nonempty(rows, field):
    for row in rows:
        value = row.get(field, "")
        if value not in ("", None):
            return str(value)
    return ""


def compare_config(rows_a, rows_b):
    mismatches = []
    for field in CONFIG_FIELDS:
        a = first_nonempty(rows_a, field)
        b = first_nonempty(rows_b, field)
        if not a or not b:
            continue
        if a != b:
            mismatches.append((field, a, b))
    return mismatches


def float_column(rows, name, default=np.nan):
    out = []
    for row in rows:
        raw = row.get(name, "")
        if raw in ("", None):
            out.append(default)
            continue
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            out.append(default)
    return np.asarray(out, dtype=np.float64)


def first_existing(fieldnames, names):
    fields = set(fieldnames)
    for name in names:
        if name in fields:
            return name
    return None


def policy_order(rows):
    for row in rows:
        value = (row.get("policy_joint_order") or "").strip()
        if value:
            joints = [part.strip() for part in value.split(",") if part.strip()]
            if len(joints) == 12:
                return joints
    return list(EXPECTED_POLICY_JOINT_ORDER)


def column_for(fieldnames, joint, index, aliases):
    return first_existing(fieldnames, [alias.format(joint=joint, index=index) for alias in aliases])


def amplitude(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return 0.5 * (float(np.max(values)) - float(np.min(values)))


def rms(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(values * values)))


def percent(values):
    if values is None:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(values > 0.5)) / float(values.size)


def dominant_frequency(rows, fieldnames, signal):
    time_col = first_existing(fieldnames, ["elapsed_s", "sim_time", "time_s"])
    if time_col is None:
        return math.nan
    t = float_column(rows, time_col)
    t = t - t[0]
    signal = np.asarray(signal, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(signal)
    t = t[mask]
    signal = signal[mask]
    if signal.size < 8:
        return math.nan
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        return math.nan
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    if freqs.size <= 1:
        return math.nan
    return float(freqs[int(np.argmax(np.abs(spectrum[1:])) + 1)])


def joint_metrics(rows, fieldnames, joint, index):
    actor_col = column_for(
        fieldnames,
        joint,
        index,
        ["{joint}_actor_q_target", "{joint}_q_actor_target", "q_actor_target_{index:02d}"],
    )
    transmitted_col = column_for(
        fieldnames,
        joint,
        index,
        ["{joint}_q_des_transmitted", "{joint}_q_target", "q_target_{index:02d}"],
    )
    measured_col = column_for(fieldnames, joint, index, ["{joint}_q_fb", "q_{index:02d}"])
    torque_col = column_for(
        fieldnames,
        joint,
        index,
        ["{joint}_measured_torque", "{joint}_fb_joint_torque", "{joint}_fb_torque"],
    )
    est_torque_col = column_for(
        fieldnames,
        joint,
        index,
        ["{joint}_estimated_pd_torque", "{joint}_cmd_tau_pd_est"],
    )
    torque_limit_col = column_for(
        fieldnames,
        joint,
        index,
        ["{joint}_torque_limit_active", "{joint}_torque_limited", "torque_clip_count_{index:02d}"],
    )
    joint_limit_col = column_for(
        fieldnames,
        joint,
        index,
        ["{joint}_joint_limit_active", "target_joint_limited_{index:02d}"],
    )
    required = {
        "actor": actor_col,
        "transmitted": transmitted_col,
        "measured": measured_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise SystemExit(f"ERROR: {joint} is missing columns: " + ", ".join(missing))

    actor = float_column(rows, actor_col)
    transmitted = float_column(rows, transmitted_col)
    measured = float_column(rows, measured_col)
    measured_torque = None if torque_col is None else float_column(rows, torque_col)
    est_torque = None if est_torque_col is None else float_column(rows, est_torque_col)
    torque_active = None if torque_limit_col is None else float_column(rows, torque_limit_col)
    joint_active = None if joint_limit_col is None else float_column(rows, joint_limit_col)
    tracking = transmitted - measured
    return {
        "actor_amp": amplitude(actor),
        "transmitted_amp": amplitude(transmitted),
        "measured_amp": amplitude(measured),
        "tracking_rms": rms(tracking),
        "tracking_max": float(np.nanmax(np.abs(tracking))) if np.any(np.isfinite(tracking)) else math.nan,
        "torque_limit_pct": percent(torque_active),
        "joint_limit_pct": percent(joint_active),
        "measured_torque_peak": (
            float(np.nanmax(np.abs(measured_torque)))
            if measured_torque is not None and np.any(np.isfinite(measured_torque))
            else math.nan
        ),
        "estimated_torque_peak": (
            float(np.nanmax(np.abs(est_torque)))
            if est_torque is not None and np.any(np.isfinite(est_torque))
            else math.nan
        ),
        "dominant_frequency": dominant_frequency(rows, fieldnames, transmitted),
    }


def summarize_run(rows, fieldnames):
    joints = policy_order(rows)
    return {
        joint: joint_metrics(rows, fieldnames, joint, index)
        for index, joint in enumerate(joints)
    }


def timing_summary(rows):
    result = {}
    for field in (
        "cycle_work_ms",
        "imu_cache_read_ms",
        "imu_sample_age_ms",
        "policy_inference_ms",
        "can_tx_ms",
        "feedback_age_max_ms",
    ):
        values = float_column(rows, field)
        values = values[np.isfinite(values)]
        if values.size:
            result[f"{field}_median"] = float(np.median(values))
            result[f"{field}_p99"] = float(np.percentile(values, 99))
    return result


def finite_min(rows, fieldnames, aliases):
    column = first_existing(fieldnames, aliases)
    if column is None:
        return math.nan
    values = float_column(rows, column)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.min(values))


def phase_degrees(rows, fieldnames, joint_a, joint_b, index_a, index_b):
    col_a = column_for(
        fieldnames,
        joint_a,
        index_a,
        ["{joint}_q_des_transmitted", "{joint}_q_target", "q_target_{index:02d}"],
    )
    col_b = column_for(
        fieldnames,
        joint_b,
        index_b,
        ["{joint}_q_des_transmitted", "{joint}_q_target", "q_target_{index:02d}"],
    )
    time_col = first_existing(fieldnames, ["elapsed_s", "sim_time", "time_s"])
    if col_a is None or col_b is None or time_col is None:
        return math.nan
    a = float_column(rows, col_a)
    b = float_column(rows, col_b)
    t = float_column(rows, time_col)
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
    a = a[mask]
    b = b[mask]
    t = t[mask]
    if a.size < 16:
        return math.nan
    dt = float(np.median(np.diff(t)))
    freq = dominant_frequency(rows, fieldnames, a)
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(freq) or freq <= 0.0:
        return math.nan
    a = a - np.mean(a)
    b = b - np.mean(b)
    if np.max(np.abs(a)) < 1.0e-6 or np.max(np.abs(b)) < 1.0e-6:
        return math.nan
    corr = np.correlate(a, b, mode="full")
    lag_samples = int(np.argmax(corr) - (a.size - 1))
    phase = float(lag_samples) * dt * freq * 360.0
    while phase > 180.0:
        phase -= 360.0
    while phase < -180.0:
        phase += 360.0
    return phase


def diagonal_phase_summary(rows, fieldnames):
    joints = policy_order(rows)
    index_by_joint = {joint: index for index, joint in enumerate(joints)}
    pairs = [
        ("FL_calf_joint", "BR_calf_joint"),
        ("FR_calf_joint", "BL_calf_joint"),
        ("FL_thigh_joint", "BR_thigh_joint"),
        ("FR_thigh_joint", "BL_thigh_joint"),
    ]
    result = {}
    for joint_a, joint_b in pairs:
        if joint_a not in index_by_joint or joint_b not in index_by_joint:
            continue
        result[f"{joint_a}_vs_{joint_b}_phase_deg"] = phase_degrees(
            rows,
            fieldnames,
            joint_a,
            joint_b,
            index_by_joint[joint_a],
            index_by_joint[joint_b],
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows_a, fields_a = load_rows(args.run_a)
    rows_b, fields_b = load_rows(args.run_b)
    mismatches = compare_config(rows_a, rows_b)
    if mismatches:
        print("ERROR: refusing comparison because run configuration differs:")
        for field, a, b in mismatches:
            print(f"  {field}: A={a!r} B={b!r}")
        return 2

    summary_a = summarize_run(rows_a, fields_a)
    summary_b = summarize_run(rows_b, fields_b)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "comparison_summary.csv"

    fieldnames = [
        "joint",
        "run_a_actor_amp",
        "run_b_actor_amp",
        "run_a_transmitted_amp",
        "run_b_transmitted_amp",
        "transmitted_amp_improvement_pct",
        "run_a_measured_amp",
        "run_b_measured_amp",
        "run_a_tracking_rms",
        "run_b_tracking_rms",
        "run_a_torque_limit_pct",
        "run_b_torque_limit_pct",
        "run_a_measured_torque_peak",
        "run_b_measured_torque_peak",
        "run_a_dominant_frequency",
        "run_b_dominant_frequency",
    ]
    rows_out = []
    for joint in policy_order(rows_a):
        a = summary_a[joint]
        b = summary_b[joint]
        improvement = math.nan
        if a["transmitted_amp"] and np.isfinite(a["transmitted_amp"]):
            improvement = 100.0 * (b["transmitted_amp"] - a["transmitted_amp"]) / a["transmitted_amp"]
        rows_out.append(
            {
                "joint": joint,
                "run_a_actor_amp": a["actor_amp"],
                "run_b_actor_amp": b["actor_amp"],
                "run_a_transmitted_amp": a["transmitted_amp"],
                "run_b_transmitted_amp": b["transmitted_amp"],
                "transmitted_amp_improvement_pct": improvement,
                "run_a_measured_amp": a["measured_amp"],
                "run_b_measured_amp": b["measured_amp"],
                "run_a_tracking_rms": a["tracking_rms"],
                "run_b_tracking_rms": b["tracking_rms"],
                "run_a_torque_limit_pct": a["torque_limit_pct"],
                "run_b_torque_limit_pct": b["torque_limit_pct"],
                "run_a_measured_torque_peak": a["measured_torque_peak"],
                "run_b_measured_torque_peak": b["measured_torque_peak"],
                "run_a_dominant_frequency": a["dominant_frequency"],
                "run_b_dominant_frequency": b["dominant_frequency"],
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    timing_a = timing_summary(rows_a)
    timing_b = timing_summary(rows_b)
    phase_a = diagonal_phase_summary(rows_a, fields_a)
    phase_b = diagonal_phase_summary(rows_b, fields_b)
    encoder_margin_a = finite_min(
        rows_a,
        fields_a,
        ["encoder_limit_margin_rad", "encoder_limit_margin", "min_encoder_limit_margin_rad"],
    )
    encoder_margin_b = finite_min(
        rows_b,
        fields_b,
        ["encoder_limit_margin_rad", "encoder_limit_margin", "min_encoder_limit_margin_rad"],
    )
    print("Timing A:", timing_a)
    print("Timing B:", timing_b)
    print("Diagonal phase A:", phase_a)
    print("Diagonal phase B:", phase_b)
    if np.isfinite(encoder_margin_a) or np.isfinite(encoder_margin_b):
        print(
            "Encoder limit margin min:",
            {"run_a_rad": encoder_margin_a, "run_b_rad": encoder_margin_b},
        )
    print("Wrote comparison:", output_csv)

    better = [
        row for row in rows_out
        if np.isfinite(row["transmitted_amp_improvement_pct"])
        and row["transmitted_amp_improvement_pct"] > 15.0
    ]
    high_torque = [
        row for row in rows_out
        if np.isfinite(row["run_b_measured_torque_peak"])
        and row["run_b_measured_torque_peak"] > 24.0
    ]
    if better and not high_torque:
        print("Result: run B materially improved transmitted target motion without >24 Nm measured peaks.")
    elif high_torque:
        print("Result: run B has high measured-torque peaks; do not increase authority further.")
    else:
        print("Result: run B did not materially improve transmitted target motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
