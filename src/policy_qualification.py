#!/usr/bin/env python3
"""Deterministic policy replay, golden vectors, and qualification gates."""

import csv
import time
from pathlib import Path

import numpy as np
import yaml

def create_golden_vectors_from_isaac_csv(
    source_csv,
    output_path,
    policy_sha256,
):
    """Package independently logged Isaac actor inputs and outputs."""
    source_csv = Path(source_csv).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observation_fields = [f"obs_{index:03d}" for index in range(48)]
    action_field_sets = (
        [f"policy_action_{index:02d}" for index in range(12)],
        [f"action_{index:02d}" for index in range(12)],
        [f"raw_action_{index:02d}" for index in range(12)],
    )
    observations = []
    expected_actions = []
    case_names = []
    with source_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if not set(observation_fields).issubset(fields):
            raise ValueError(
                "Isaac CSV must contain obs_000 through obs_047 exactly as "
                "passed to the actor"
            )
        action_fields = next(
            (candidate for candidate in action_field_sets if set(candidate).issubset(fields)),
            None,
        )
        if action_fields is None:
            raise ValueError(
                "Isaac CSV must contain policy_action_00..11, action_00..11, "
                "or raw_action_00..11"
            )
        for row_index, row in enumerate(reader):
            try:
                observation = np.asarray(
                    [float(row[name]) for name in observation_fields],
                    dtype=np.float32,
                )
                action = np.asarray(
                    [float(row[name]) for name in action_fields],
                    dtype=np.float32,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(action)):
                continue
            observations.append(observation)
            expected_actions.append(action)
            case_names.append(str(row.get("step", row_index)))
    if not observations:
        raise ValueError(f"No complete finite Isaac actor rows found in {source_csv}")

    np.savez_compressed(
        output_path,
        case_names=np.asarray(case_names),
        observations=np.asarray(observations, dtype=np.float32),
        expected_actions=np.asarray(expected_actions, dtype=np.float32),
        policy_sha256=np.asarray(str(policy_sha256)),
        source_csv=np.asarray(str(source_csv)),
    )
    return output_path


def check_golden_vectors(runner, input_path, tolerance=1.0e-5):
    input_path = Path(input_path).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        expected = np.asarray(data["expected_actions"], dtype=np.float32)
        policy_hash = str(np.asarray(data["policy_sha256"]).item())

    errors = []
    if observations.ndim != 2 or observations.shape[1] != 48:
        errors.append(f"golden observations shape is {observations.shape}, expected [N, 48]")
    if expected.shape != (observations.shape[0], 12):
        errors.append(f"golden actions shape is {expected.shape}, expected [N, 12]")
    if policy_hash != runner.policy_sha256:
        errors.append(f"policy SHA256 differs: golden={policy_hash} loaded={runner.policy_sha256}")
    if errors:
        return {
            "passed": False,
            "errors": errors,
            "maximum_absolute_error": float("inf"),
            "case_count": int(observations.shape[0]) if observations.ndim else 0,
        }

    actual = np.stack([runner.infer_action(obs) for obs in observations])
    maximum_error = float(np.max(np.abs(actual - expected)))
    if not np.all(np.isfinite(actual)):
        errors.append("policy returned NaN or Inf")
    if maximum_error > float(tolerance):
        errors.append(
            f"maximum absolute output difference {maximum_error:.9g} exceeds {tolerance:.9g}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "maximum_absolute_error": maximum_error,
        "case_count": int(observations.shape[0]),
    }


def _float_cell(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"CSV row is missing finite numeric field {name}")


def read_policy_replay_rows(csv_path):
    csv_path = Path(csv_path).expanduser().resolve()
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            if not source.get("obs_000", "").strip():
                continue
            observation = np.asarray(
                [_float_cell(source, f"obs_{index:03d}") for index in range(48)],
                dtype=np.float32,
            )
            original = None
            if all(source.get(f"action_{index:02d}", "").strip() for index in range(12)):
                original = np.asarray(
                    [_float_cell(source, f"action_{index:02d}") for index in range(12)],
                    dtype=np.float32,
                )
            timestamp = source.get("elapsed_s", source.get("sim_time", ""))
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError):
                timestamp = None
            rows.append((timestamp, observation, original))
    if not rows:
        raise ValueError(f"No complete 48D observation rows found in {csv_path}")
    return rows


def replay_policy_csv(runner, csv_path, output_path=None, fixed_50_hz=False, realtime=False):
    rows = read_policy_replay_rows(csv_path)
    outputs = []
    maximum_error = 0.0
    output_rows = []
    previous_timestamp = None
    deadline = time.monotonic()
    for index, (timestamp, observation, original) in enumerate(rows):
        if realtime and index:
            if fixed_50_hz or timestamp is None or previous_timestamp is None:
                delay = 0.02
            else:
                delay = max(0.0, min(1.0, timestamp - previous_timestamp))
            deadline += delay
            time.sleep(max(0.0, deadline - time.monotonic()))
        output = runner.infer_action(observation)
        outputs.append(output)
        error = float("nan")
        if original is not None:
            error = float(np.max(np.abs(output - original)))
            maximum_error = max(maximum_error, error)
        output_rows.append((index, timestamp, error, observation, output))
        previous_timestamp = timestamp

    if output_path is None:
        source = Path(csv_path).expanduser().resolve()
        output_path = source.with_name(source.stem + "_policy_replay.csv")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["replay_index", "source_timestamp", "max_abs_action_error"]
    fields += [f"obs_{index:03d}" for index in range(48)]
    fields += [f"action_{index:02d}" for index in range(12)]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, timestamp, error, observation, output in output_rows:
            record = {
                "replay_index": index,
                "source_timestamp": "" if timestamp is None else timestamp,
                "max_abs_action_error": "" if not np.isfinite(error) else error,
            }
            record.update({f"obs_{i:03d}": float(v) for i, v in enumerate(observation)})
            record.update({f"action_{i:02d}": float(v) for i, v in enumerate(output)})
            writer.writerow(record)
    return {
        "output_path": output_path,
        "row_count": len(outputs),
        "maximum_absolute_error": maximum_error,
        "actions": np.asarray(outputs, dtype=np.float32),
    }


def calf_calibration_gate(path, joint_name="FL_calf_joint"):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return False, f"missing calibration recommendation: {path}"
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        entry = (data.get("calf_endpoint_calibration", {}) or {}).get(joint_name, {})
        passed = bool(entry.get("passed", False))
        repeatability = float(entry.get("maximum_repeatability_rad", float("inf")))
        if not passed:
            return False, f"{joint_name} calibration is not marked passed"
        if not np.isfinite(repeatability) or repeatability > 0.02:
            return False, f"{joint_name} repeatability {repeatability:.4f} rad exceeds 0.020 rad"
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return False, f"invalid calibration recommendation: {exc}"
    return True, f"{joint_name} endpoint calibration passed"


def root_cause_report_lines(results=None):
    results = dict(results or {})
    names = (
        ("Raw actor periodic gait", "raw_actor_periodic_gait"),
        ("Policy joint order", "policy_joint_order"),
        ("Motor routing", "motor_routing"),
        ("Joint velocity validation", "joint_velocity_validation"),
        ("Observation contract", "observation_contract"),
        ("Actor gait preserved after target filters", "actor_gait_preserved"),
        ("Motor tracking", "motor_tracking"),
        ("50 Hz timing", "timing_50hz"),
        ("200 Hz CAN command timing", "can_command_200hz"),
        ("Encoder calibration", "encoder_calibration"),
        ("Torque authority", "torque_authority"),
        ("Ground-contact validity", "ground_contact_validity"),
    )
    lines = ["GAIT ROOT-CAUSE REPORT"]
    for label, key in names:
        value = str(results.get(key, "UNKNOWN")).upper()
        lines.append(f"{label}: {value}")
    return lines
