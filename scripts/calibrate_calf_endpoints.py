#!/usr/bin/env python3
"""Passively capture safe calf extension, stand, and crouch endpoints."""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from can_topology import (  # noqa: E402
    add_can_topology_args,
    close_can_buses,
    open_can_buses,
    ports_for_active_joints,
    resolve_joint_can_bus,
    resolve_port_by_bus,
)
from check_motor_connections import (  # noqa: E402
    feedback_for_joint,
    load_config,
    update_feedback_from_frames,
)
from motor_command_layer import MotorCommandLayer  # noqa: E402


CALF_JOINTS = (
    "FL_calf_joint",
    "FR_calf_joint",
    "BL_calf_joint",
    "BR_calf_joint",
)
ENDPOINTS = (
    ("safe_extension", "SAFE MAXIMUM EXTENSION"),
    ("stand", "STAND POSITION"),
    ("safe_crouch", "SAFE MAXIMUM CROUCH"),
)


def summarize_samples(raw_samples, direction, offset):
    raw = np.asarray(raw_samples, dtype=np.float64)
    joint = float(direction) * (raw - float(offset))
    return {
        "sample_count": int(raw.size),
        "raw_mean_rad": float(np.mean(raw)),
        "raw_standard_deviation_rad": float(np.std(raw)),
        "raw_repeatability_rad": float(np.max(raw) - np.min(raw)),
        "joint_mean_rad": float(np.mean(joint)),
        "joint_standard_deviation_rad": float(np.std(joint)),
        "joint_repeatability_rad": float(np.max(joint) - np.min(joint)),
    }


def evaluate_joint(endpoints, configured_limits, repeatability_limit=0.02):
    values = [float(endpoints[name]["joint_mean_rad"]) for name, _ in ENDPOINTS]
    repeatability = max(
        float(endpoints[name]["joint_repeatability_rad"]) for name, _ in ENDPOINTS
    )
    lo, stand, hi = values
    monotonic = (lo <= stand <= hi) or (hi <= stand <= lo)
    measured_min = min(values)
    measured_max = max(values)
    configured_min = float(configured_limits["min"])
    configured_max = float(configured_limits["max"])
    margin = max(0.01, 0.5 * repeatability)
    reasons = []
    if not monotonic:
        reasons.append("stand is not between the two captured safe endpoints")
    if repeatability > float(repeatability_limit):
        reasons.append(
            f"endpoint repeatability {repeatability:.4f} rad exceeds "
            f"{float(repeatability_limit):.4f} rad"
        )
    if measured_min < configured_min:
        reasons.append(
            f"measured safe minimum {measured_min:+.4f} is below configured "
            f"minimum {configured_min:+.4f} by {configured_min - measured_min:.4f} rad"
        )
    if measured_max > configured_max:
        reasons.append(
            f"measured safe maximum {measured_max:+.4f} is above configured "
            f"maximum {configured_max:+.4f} by {measured_max - configured_max:.4f} rad"
        )
    return {
        "passed": bool(monotonic and repeatability <= float(repeatability_limit)),
        "maximum_repeatability_rad": repeatability,
        "measured_safe_min_rad": measured_min,
        "measured_safe_max_rad": measured_max,
        "recommended_min_rad": measured_min - margin,
        "recommended_max_rad": measured_max + margin,
        "configured_min_rad": configured_min,
        "configured_max_rad": configured_max,
        "diagnostic_reasons": reasons,
    }


def capture_samples(
    prompt,
    joint_name,
    buses,
    layer,
    poll_commands,
    active_key,
    motor_ids,
    joint_can_bus,
    feedback_by_bus_motor_id,
    sample_count,
):
    input(prompt)
    readings = []
    last_timestamp = None
    deadline = time.monotonic() + 5.0
    while len(readings) < int(sample_count) and time.monotonic() < deadline:
        layer.send_raw_commands(buses, poll_commands)
        frames = MotorCommandLayer.read_all_frames(buses, timeout=0.05)
        update_feedback_from_frames(
            frames,
            layer,
            active_key,
            feedback_by_bus_motor_id,
            {},
        )
        feedback = feedback_for_joint(
            joint_name,
            motor_ids,
            joint_can_bus,
            feedback_by_bus_motor_id,
            {},
        )
        timestamp = None if feedback is None else feedback.get("timestamp")
        if (
            feedback is not None
            and int(feedback.get("fault_bits", 0)) == 0
            and timestamp != last_timestamp
        ):
            readings.append(float(feedback["position"]))
            last_timestamp = timestamp
        time.sleep(0.005)
    if len(readings) < int(sample_count):
        raise RuntimeError(
            f"{joint_name}: received {len(readings)}/{sample_count} fresh samples"
        )
    print(
        f"  raw mean={statistics.fmean(readings):+.6f} rad "
        f"range={max(readings) - min(readings):.6f} rad"
    )
    return readings


def main():
    parser = argparse.ArgumentParser()
    add_can_topology_args(parser, default_port="slcan0", default_can_count=1)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--calf-joints", nargs="*", choices=CALF_JOINTS, default=list(CALF_JOINTS))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeatability-limit-rad", type=float, default=0.02)
    parser.add_argument("--output-dir", default=str(ROOT / "config"))
    args = parser.parse_args()
    if args.samples < 5:
        parser.error("--samples must be at least 5")

    joint_cfg, motor_cfg = load_config()
    policy_order = list(joint_cfg["policy_to_real_order"])
    motor_ids = motor_cfg["motor_ids"]
    joints = list(args.calf_joints)
    joint_can_bus = resolve_joint_can_bus(policy_order, args.can_count)
    layer = MotorCommandLayer(
        policy_order=policy_order,
        motor_ids=motor_ids,
        active_joints=joints,
        joint_can_bus=joint_can_bus,
    )
    port_by_bus = resolve_port_by_bus(args)
    buses = open_can_buses(
        ports_for_active_joints(port_by_bus, joint_can_bus, joints),
        baud=args.baud,
        timeout=0.002,
        backend=args.can_backend,
        bitrate=args.can_bitrate,
    )
    poll_commands = layer.build_feedback_poll_commands()
    stop_commands = layer.build_stop_commands()
    layer.send_raw_commands(buses, stop_commands)
    feedback_by_bus_motor_id = {}
    limit_cfg = yaml.safe_load((ROOT / "config" / "joint_limits.yaml").read_text())[
        "joint_limits"
    ]
    recommendation = {"calf_endpoint_calibration": {}}
    report = [
        "CALF ENDPOINT CALIBRATION REPORT",
        "Motor state: passive/disabled; this tool never sends movement commands.",
        "",
    ]
    try:
        for joint_name in joints:
            print(f"\n{joint_name}: motor remains disabled")
            active_key = {(joint_can_bus[joint_name], int(motor_ids[joint_name]))}
            endpoints = {}
            for endpoint_name, endpoint_label in ENDPOINTS:
                samples = capture_samples(
                    f"Move {joint_name} to {endpoint_label}, hold it, then press Enter: ",
                    joint_name,
                    buses,
                    layer,
                    poll_commands,
                    active_key,
                    motor_ids,
                    joint_can_bus,
                    feedback_by_bus_motor_id,
                    args.samples,
                )
                endpoints[endpoint_name] = summarize_samples(
                    samples,
                    layer.joint_directions[joint_name],
                    layer.joint_offsets[joint_name],
                )
            result = evaluate_joint(
                endpoints,
                limit_cfg[joint_name],
                repeatability_limit=args.repeatability_limit_rad,
            )
            entry = {
                "motor_id": int(motor_ids[joint_name]),
                "motor_direction": float(layer.joint_directions[joint_name]),
                "configured_encoder_offset_rad": float(layer.joint_offsets[joint_name]),
                **result,
                "endpoints": endpoints,
            }
            recommendation["calf_endpoint_calibration"][joint_name] = entry
            report.extend([
                joint_name,
                f"  result: {'PASS' if result['passed'] else 'FAIL'}",
                f"  measured range: [{result['measured_safe_min_rad']:+.6f}, "
                f"{result['measured_safe_max_rad']:+.6f}] rad",
                f"  configured range: [{result['configured_min_rad']:+.6f}, "
                f"{result['configured_max_rad']:+.6f}] rad",
                f"  recommended review range: [{result['recommended_min_rad']:+.6f}, "
                f"{result['recommended_max_rad']:+.6f}] rad",
            ])
            if result["diagnostic_reasons"]:
                report.extend(f"  reason: {reason}" for reason in result["diagnostic_reasons"])
            else:
                report.append("  reason: repeatable endpoints and monotonic stand placement")
            report.append("")
    finally:
        try:
            layer.send_raw_commands(buses, stop_commands)
        finally:
            close_can_buses(buses)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "calf_endpoint_recommendation.yaml"
    report_path = output_dir / "calf_calibration_report.txt"
    yaml_path.write_text(yaml.safe_dump(recommendation, sort_keys=False), encoding="utf-8")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nRecommendation only; active joint limits were not modified.")
    print("Recommendation:", yaml_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
