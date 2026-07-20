#!/usr/bin/env python3
"""Generate target, feedback and torque plots from a Grallator run CSV."""

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


LEG_JOINTS = {
    "FL": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
    "FR": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
    "BL": ["BL_hip_joint", "BL_thigh_joint", "BL_calf_joint"],
    "BR": ["BR_hip_joint", "BR_thigh_joint", "BR_calf_joint"],
}


def load_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise SystemExit(f"ERROR: {path} has no data rows")
    return rows, fieldnames


def float_column(rows, name, default=np.nan):
    values = []
    for row in rows:
        raw = row.get(name, "")
        if raw in ("", None):
            values.append(default)
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(default)
    return np.asarray(values, dtype=np.float64)


def first_existing(fieldnames, names):
    fieldset = set(fieldnames)
    for name in names:
        if name in fieldset:
            return name
    return None


def policy_order_from_rows(rows):
    for row in rows:
        value = (row.get("policy_joint_order") or "").strip()
        if value:
            joints = [part.strip() for part in value.split(",") if part.strip()]
            if len(joints) == 12:
                return joints
    return list(EXPECTED_POLICY_JOINT_ORDER)


def time_axis(rows, fieldnames):
    name = first_existing(fieldnames, ["elapsed_s", "sim_time", "time_s", "step"])
    if name is None:
        raise SystemExit("ERROR: CSV needs one of elapsed_s, sim_time, time_s or step")
    values = float_column(rows, name)
    if name == "step":
        return values, "step"
    return values - values[0], "time_s"


def joint_column(fieldnames, joint, index, aliases):
    candidates = []
    for template in aliases:
        candidates.append(template.format(joint=joint, index=index))
    return first_existing(fieldnames, candidates)


def joint_series(rows, fieldnames, joint, index):
    columns = {
        "actor": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_actor_q_target", "{joint}_q_actor_target", "q_actor_target_{index:02d}"],
        ),
        "entry": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_entry_blended_q_target", "q_entry_blended_target_{index:02d}"],
        ),
        "limit": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_joint_limit_filtered_q_target", "q_joint_limit_filtered_target_{index:02d}"],
        ),
        "rate": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_rate_limited_q_target", "q_rate_limited_target_{index:02d}"],
        ),
        "transmitted": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_q_des_transmitted", "{joint}_q_target", "q_target_{index:02d}"],
        ),
        "measured": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_q_fb", "q_{index:02d}"],
        ),
        "estimated_torque": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_estimated_pd_torque", "{joint}_cmd_tau_pd_est"],
        ),
        "measured_torque": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_measured_torque", "{joint}_fb_joint_torque", "{joint}_fb_torque"],
        ),
        "joint_limit_active": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_joint_limit_active", "target_joint_limited_{index:02d}"],
        ),
        "rate_limit_active": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_rate_limit_active", "target_rate_limited_{index:02d}"],
        ),
        "torque_limit_active": joint_column(
            fieldnames,
            joint,
            index,
            ["{joint}_torque_limit_active", "{joint}_torque_limited"],
        ),
    }
    required = ["actor", "entry", "transmitted", "measured"]
    missing = [name for name in required if columns[name] is None]
    if missing:
        raise SystemExit(
            f"ERROR: missing required target/feedback column(s) for {joint}: "
            + ", ".join(missing)
        )
    return {
        key: None if column is None else float_column(rows, column)
        for key, column in columns.items()
    }


def amplitude(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return 0.5 * (float(np.nanmax(values)) - float(np.nanmin(values)))


def rms(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(values * values)))


def dominant_frequency(time_s, signal):
    time_s = np.asarray(time_s, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    mask = np.isfinite(time_s) & np.isfinite(signal)
    time_s = time_s[mask]
    signal = signal[mask]
    if signal.size < 8:
        return math.nan
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0.0:
        return math.nan
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    if freqs.size <= 1:
        return math.nan
    peak = int(np.argmax(np.abs(spectrum[1:])) + 1)
    return float(freqs[peak])


def percent_active(values):
    if values is None:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(values > 0.5)) / float(values.size)


def save_joint_plot(plt, output_dir, t, joint, series):
    plt.figure(figsize=(10, 5))
    plt.plot(t, series["actor"], label="actor target", linewidth=1.3)
    plt.plot(t, series["entry"], label="entry-blended target", linewidth=1.1)
    if series.get("limit") is not None:
        plt.plot(t, series["limit"], label="joint-limit output", linewidth=0.9)
    if series.get("rate") is not None:
        plt.plot(t, series["rate"], label="rate-limit output", linewidth=0.9)
    plt.plot(t, series["transmitted"], label="transmitted q_des", linewidth=1.2)
    plt.plot(t, series["measured"], label="measured q", linewidth=1.0)
    plt.title(joint)
    plt.xlabel("time (s)")
    plt.ylabel("joint position (rad)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{joint}.png", dpi=140)
    plt.close()


def save_torque_plot(plt, output_dir, t, joint, series, torque_limit):
    if series.get("estimated_torque") is None and series.get("measured_torque") is None:
        return
    plt.figure(figsize=(10, 4))
    if series.get("estimated_torque") is not None:
        plt.plot(t, series["estimated_torque"], label="estimated PD torque", linewidth=1.1)
    if series.get("measured_torque") is not None:
        plt.plot(t, series["measured_torque"], label="measured torque", linewidth=1.1)
    if torque_limit is not None:
        plt.axhline(float(torque_limit), color="tab:red", linestyle="--", linewidth=0.9)
        plt.axhline(-float(torque_limit), color="tab:red", linestyle="--", linewidth=0.9)
    active = series.get("torque_limit_active")
    if active is not None:
        active = np.asarray(active, dtype=np.float64) > 0.5
        if np.any(active):
            ymin, ymax = plt.ylim()
            plt.fill_between(t, ymin, ymax, where=active, color="tab:red", alpha=0.12, label="torque limited")
    plt.title(f"{joint} torque")
    plt.xlabel("time (s)")
    plt.ylabel("torque (Nm)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{joint}_torque.png", dpi=140)
    plt.close()


def save_leg_plot(plt, output_dir, t, leg, all_series):
    plt.figure(figsize=(11, 6))
    for joint in LEG_JOINTS[leg]:
        series = all_series.get(joint)
        if series is None:
            continue
        plt.plot(t, series["transmitted"], label=f"{joint} q_des", linewidth=1.0)
        plt.plot(t, series["measured"], label=f"{joint} q", linewidth=0.9, alpha=0.75)
    plt.title(f"{leg} leg")
    plt.xlabel("time (s)")
    plt.ylabel("joint position (rad)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{leg}_leg.png", dpi=140)
    plt.close()


def save_diagonal_plot(plt, output_dir, t, name, leg_a, leg_b, all_series):
    plt.figure(figsize=(11, 6))
    for suffix in ("hip_joint", "thigh_joint", "calf_joint"):
        joint_a = f"{leg_a}_{suffix}"
        joint_b = f"{leg_b}_{suffix}"
        if joint_a in all_series and joint_b in all_series:
            plt.plot(t, all_series[joint_a]["transmitted"], label=f"{joint_a} q_des")
            plt.plot(t, all_series[joint_b]["transmitted"], label=f"{joint_b} q_des", linestyle="--")
    plt.title(name)
    plt.xlabel("time (s)")
    plt.ylabel("transmitted q_des (rad)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name.replace(' ', '_')}.png", dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--torque-limit", type=float, default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    rows, fieldnames = load_rows(args.csv)
    joints = policy_order_from_rows(rows)
    t, _time_label = time_axis(rows, fieldnames)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_series = {}
    summary_rows = []
    for index, joint in enumerate(joints):
        series = joint_series(rows, fieldnames, joint, index)
        all_series[joint] = series
        save_joint_plot(plt, output_dir, t, joint, series)
        save_torque_plot(plt, output_dir, t, joint, series, args.torque_limit)

        actor_amp = amplitude(series["actor"])
        transmitted_amp = amplitude(series["transmitted"])
        measured_amp = amplitude(series["measured"])
        tracking = series["transmitted"] - series["measured"]
        summary_rows.append(
            {
                "joint": joint,
                "target_amplitude": actor_amp,
                "transmitted_amplitude": transmitted_amp,
                "measured_amplitude": measured_amp,
                "target_to_transmitted_amplitude_ratio": (
                    transmitted_amp / actor_amp if actor_amp and np.isfinite(actor_amp) else math.nan
                ),
                "transmitted_to_measured_amplitude_ratio": (
                    measured_amp / transmitted_amp
                    if transmitted_amp and np.isfinite(transmitted_amp)
                    else math.nan
                ),
                "rms_tracking_error": rms(tracking),
                "torque_limit_percentage": percent_active(series.get("torque_limit_active")),
                "joint_limit_percentage": percent_active(series.get("joint_limit_active")),
                "dominant_gait_frequency": dominant_frequency(t, series["transmitted"]),
            }
        )

    for leg in ("FL", "FR", "BL", "BR"):
        save_leg_plot(plt, output_dir, t, leg, all_series)
    save_diagonal_plot(plt, output_dir, t, "FL versus BR", "FL", "BR", all_series)
    save_diagonal_plot(plt, output_dir, t, "FR versus BL", "FR", "BL", all_series)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Wrote gait plots:", output_dir)
    print("Wrote summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
