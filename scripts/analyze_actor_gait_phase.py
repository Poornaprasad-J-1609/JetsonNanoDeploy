#!/usr/bin/env python3
"""Locate the stage where a policy gait loses periodic diagonal structure."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gait_phase_analysis import classify_diagonal_trot  # noqa: E402
from joint_mapping import POLICY_JOINT_ORDER  # noqa: E402


def _value(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return np.nan


def _load(path, steady_only):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("action_00", "") != ""]
    if steady_only:
        rows = [
            row for row in rows
            if _value(row, "policy_entry_scale") >= 0.999 and row.get("mode") == "policy"
        ]
    if len(rows) < 20:
        raise ValueError("need at least 20 policy rows for gait-phase analysis")
    return rows


def _array(rows, keys):
    return np.asarray([[_value(row, key) for key in keys] for row in rows], dtype=np.float64)


def _write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_csv(path, steady_only=False):
    rows = _load(path, steady_only)
    elapsed = np.asarray([_value(row, "elapsed_s") for row in rows])
    dt = np.diff(elapsed)
    valid_dt = dt[np.isfinite(dt) & (dt > 1.0e-4) & (dt < 0.2)]
    sample_rate = 50.0 if valid_dt.size == 0 else 1.0 / float(np.median(valid_dt))
    signs_cfg = yaml.safe_load((ROOT / "config" / "joint_map.yaml").read_text())
    sign_map = signs_cfg.get("policy_joint_signs", {}) or {}
    signs = np.asarray([float(sign_map.get(name, 1.0)) for name in POLICY_JOINT_ORDER])

    stages = {
        "raw_actor": _array(rows, [f"action_{i:02d}" for i in range(12)]) * signs,
        "actor_target": _array(rows, [f"q_actor_target_{i:02d}" for i in range(12)]) * signs,
        "transmitted_target": _array(rows, [f"q_target_{i:02d}" for i in range(12)]) * signs,
        "measured": _array(rows, [f"q_{i:02d}" for i in range(12)]) * signs,
    }
    results = {}
    for name, values in stages.items():
        minimum_amplitude = 0.03 if name == "raw_actor" else 0.008
        status, estimates, phase_rows = classify_diagonal_trot(
            values,
            sample_rate,
            minimum_amplitude=minimum_amplitude,
        )
        results[name] = {
            "status": status,
            "estimates": estimates,
            "phase_rows": phase_rows,
        }
    return sample_rate, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--steady-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sample_rate, results = analyze_csv(args.csv, steady_only=args.steady_only)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "raw_actor": "raw_actor_phase_matrix.csv",
        "actor_target": "actor_target_phase_matrix.csv",
        "transmitted_target": "transmitted_target_phase_matrix.csv",
        "measured": "measured_phase_matrix.csv",
    }
    for name, filename in filenames.items():
        _write_rows(output_dir / filename, results[name]["phase_rows"])

    raw = results["raw_actor"]["status"]
    transmitted = results["transmitted_target"]["status"]
    measured = results["measured"]["status"]
    if raw != "PASS":
        lost = "raw actor"
    elif transmitted != "PASS":
        lost = "deployment target filters/torque limiting"
    elif measured != "PASS":
        lost = "motor tracking or control timing"
    else:
        lost = "not lost in the logged command/measurement pipeline"
    lines = [
        "GAIT PHASE REPORT",
        f"samples_per_second: {sample_rate:.3f}",
        f"Raw actor periodic gait: {raw}",
        f"Actor joint target periodic gait: {results['actor_target']['status']}",
        f"Transmitted target periodic gait: {transmitted}",
        f"Measured motion periodic gait: {measured}",
        f"Pipeline stage where gait is lost: {lost}",
        "",
        f"Does the raw actor contain a recognizable gait? {raw}",
        f"Does the transmitted target preserve that gait? {transmitted}",
        f"Does measured motion preserve that gait? {measured}",
        f"At which pipeline stage is the gait lost? {lost}",
    ]
    for stage_name, result in results.items():
        lines.append(f"\n{stage_name} joints:")
        for joint_name, estimate in zip(POLICY_JOINT_ORDER, result["estimates"]):
            lines.append(
                f"  {joint_name}: frequency={estimate.frequency_hz:.3f}Hz "
                f"amplitude={estimate.amplitude:.4f} phase={np.degrees(estimate.phase_rad):+.1f}deg "
                f"periodic={estimate.periodic} sufficient_amplitude={estimate.sufficient_amplitude}"
            )
    report = "\n".join(lines) + "\n"
    print(report, end="")
    (output_dir / "gait_phase_report.txt").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
