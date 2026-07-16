#!/usr/bin/env python3
"""Capture four-bar motor-angle calibration from RobStride encoder feedback.

This script never enables MIT control and never sends position commands. For
each requested knee angle, manually place the calf linkage at that measured
angle, press Enter, and the script records sign/offset-corrected motor angles
from the encoder feedback. It then fills config/four_bar_transmission.yaml.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from can_topology import (
    add_can_topology_args,
    close_can_buses,
    open_can_buses,
    ports_for_active_joints,
    resolve_joint_can_bus,
    resolve_port_by_bus,
)
from motor_command_layer import (
    MotorCommandLayer,
    decode_mit_feedback_frame,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALF_JOINTS = (
    "FR_calf_joint",
    "FL_calf_joint",
    "BR_calf_joint",
    "BL_calf_joint",
)
DEFAULT_KNEE_ANGLES = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def save_yaml(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    tmp.replace(path)


def load_config():
    joint_cfg = load_yaml(ROOT / "config" / "joint_map.yaml")
    motor_cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    return joint_cfg, motor_cfg


def feedback_for_active_frames(frames, layer, active_bus_motor_keys):
    by_key = {}
    for frame in frames:
        feedback = decode_mit_feedback_frame(frame.can_id, frame.data, layer.proto)
        if feedback is None:
            continue
        bus_name = getattr(frame, "bus_name", None)
        if bus_name is None:
            continue
        motor_id = int(feedback["motor_id"])
        key = (bus_name, motor_id)
        if key in active_bus_motor_keys:
            feedback = dict(feedback)
            feedback["timestamp"] = float(getattr(frame, "timestamp", time.monotonic()))
            feedback["bus_name"] = bus_name
            by_key[key] = feedback
    return by_key


def collect_motor_theta_samples(
    buses,
    layer,
    motor_ids,
    joint_can_bus,
    joints,
    sample_seconds,
    rate_hz,
):
    poll_commands = layer.build_feedback_poll_commands()
    active_bus_motor_keys = {
        (joint_can_bus.get(name, "can0"), int(motor_ids[name]))
        for name in joints
    }
    samples = {joint_name: [] for joint_name in joints}
    deadline = time.monotonic() + float(sample_seconds)
    dt = 1.0 / max(1e-6, float(rate_hz))

    while time.monotonic() < deadline:
        layer.send_raw_commands(buses, poll_commands)
        frames = layer.read_all_frames(buses, timeout=min(0.05, dt))
        by_key = feedback_for_active_frames(frames, layer, active_bus_motor_keys)
        for joint_name in joints:
            key = (joint_can_bus.get(joint_name, "can0"), int(motor_ids[joint_name]))
            feedback = by_key.get(key)
            if feedback is None:
                continue
            if int(feedback.get("fault_bits", 0)) != 0:
                continue
            raw = float(feedback["position"])
            offset = float(layer.joint_offsets[joint_name])
            direction = float(layer.joint_directions[joint_name])
            motor_theta = direction * (raw - offset)
            samples[joint_name].append(motor_theta)
        time.sleep(max(0.0, dt - 0.001 * len(poll_commands)))

    missing = [joint_name for joint_name, values in samples.items() if not values]
    if missing:
        raise RuntimeError(
            "No fresh encoder samples for: " + ", ".join(missing)
        )
    return {
        joint_name: float(statistics.median(values))
        for joint_name, values in samples.items()
    }


def profile_name_for_joint(joint_name: str) -> str:
    return str(joint_name).replace("_joint", "")


def ensure_four_bar_yaml_shape(cfg, joints):
    root = cfg.setdefault("four_bar_transmission", {})
    root.setdefault("enabled", False)
    root.setdefault("require_feedback_for_commands", True)
    root.setdefault("clamp_policy_to_hard_limits", True)
    profiles = root.setdefault("profiles", {})
    joint_cfg = root.setdefault("joints", {})
    for joint_name in joints:
        profile_name = profile_name_for_joint(joint_name)
        profiles.setdefault(
            profile_name,
            {
                "motor_angle_rad": [],
                "knee_angle_rad": [],
                "efficiency": 1.0,
                "motor_torque_limit_nm": 120.0,
                "min_abs_jacobian": 0.05,
                "clamp_outside_calibration": False,
                "compensate_efficiency_in_commands": True,
            },
        )
        existing = profiles[profile_name]
        existing.setdefault("efficiency", 1.0)
        existing.setdefault("motor_torque_limit_nm", 120.0)
        existing.setdefault("min_abs_jacobian", 0.05)
        existing.setdefault("clamp_outside_calibration", False)
        existing.setdefault("compensate_efficiency_in_commands", True)
        virtual_sign = -1 if joint_name.startswith(("FR_", "BR_")) else 1
        joint_cfg[joint_name] = {
            "enabled": True,
            "profile": profile_name,
            "virtual_sign": virtual_sign,
        }
    return root


def validate_monotonic(values, label):
    if len(values) < 3:
        raise ValueError(f"{label} needs at least 3 points")
    diffs = [b - a for a, b in zip(values, values[1:])]
    increasing = all(diff > 0.0 for diff in diffs)
    decreasing = all(diff < 0.0 for diff in diffs)
    if not (increasing or decreasing):
        raise ValueError(
            f"{label} must be strictly monotonic; captured values={values}"
        )


def update_four_bar_yaml(path, joints, knee_angles, captures, enable_after):
    cfg = load_yaml(path)
    root = ensure_four_bar_yaml_shape(cfg, joints)
    for joint_name in joints:
        profile_name = root["joints"][joint_name]["profile"]
        motor_angles = [float(capture[joint_name]) for capture in captures]
        validate_monotonic(motor_angles, f"{profile_name}.motor_angle_rad")
        validate_monotonic(knee_angles, f"{profile_name}.knee_angle_rad")
        profile = root["profiles"][profile_name]
        profile["motor_angle_rad"] = [round(value, 6) for value in motor_angles]
        profile["knee_angle_rad"] = [round(float(value), 6) for value in knee_angles]

    root["enabled"] = bool(enable_after)
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    save_yaml(path, cfg)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_can_topology_args(parser, default_port="slcan0", default_can_count=1)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--joints",
        nargs="+",
        default=list(DEFAULT_CALF_JOINTS),
        help="calf joints to calibrate",
    )
    parser.add_argument(
        "--knee-angles",
        nargs="+",
        type=float,
        default=list(DEFAULT_KNEE_ANGLES),
        help="measured positive knee flexion angles; manually move to each value",
    )
    parser.add_argument("--sample-seconds", type=float, default=0.8)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument(
        "--enable-after",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="set four_bar_transmission.enabled after writing the table",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="do not ask for final confirmation before writing YAML",
    )
    args = parser.parse_args()

    if len(args.knee_angles) < 3:
        parser.error("--knee-angles needs at least 3 values")

    joint_cfg, motor_cfg = load_config()
    policy_order = list(joint_cfg["policy_to_real_order"])
    joints = list(args.joints)
    unknown = [joint for joint in joints if joint not in policy_order]
    if unknown:
        parser.error("unknown joint(s): " + ", ".join(unknown))
    missing_ids = [joint for joint in joints if joint not in motor_cfg["motor_ids"]]
    if missing_ids:
        parser.error("missing motor IDs for: " + ", ".join(missing_ids))

    port_by_bus = resolve_port_by_bus(args)
    joint_can_bus = resolve_joint_can_bus(policy_order, args.can_count)
    layer = MotorCommandLayer(
        policy_order=policy_order,
        motor_ids=motor_cfg["motor_ids"],
        active_joints=joints,
        joint_can_bus=joint_can_bus,
    )
    active_port_by_bus = ports_for_active_joints(
        port_by_bus,
        joint_can_bus,
        layer.active_joints,
    ) or port_by_bus

    print("Four-bar encoder calibration capture")
    print("No MIT enable or motion commands are sent. Only feedback poll frames are used.")
    print("Joints:", ", ".join(joints))
    print("Knee angles:", ", ".join(f"{value:.3f}" for value in args.knee_angles))
    print()

    buses = open_can_buses(
        active_port_by_bus,
        baud=args.baud,
        timeout=0.002,
        backend=args.can_backend,
        bitrate=args.can_bitrate,
    )

    captures = []
    try:
        for index, knee_angle in enumerate(args.knee_angles, start=1):
            input(
                f"[{index}/{len(args.knee_angles)}] Manually set measured knee "
                f"flexion to {knee_angle:+.4f} rad, then press Enter..."
            )
            capture = collect_motor_theta_samples(
                buses=buses,
                layer=layer,
                motor_ids=motor_cfg["motor_ids"],
                joint_can_bus=joint_can_bus,
                joints=joints,
                sample_seconds=args.sample_seconds,
                rate_hz=args.rate,
            )
            captures.append(capture)
            for joint_name in joints:
                print(f"  {joint_name:20s} motor_angle_rad={capture[joint_name]:+.6f}")
            print()
    finally:
        close_can_buses(buses)

    print("Captured table:")
    for joint_name in joints:
        values = [capture[joint_name] for capture in captures]
        print(
            f"  {profile_name_for_joint(joint_name):10s} "
            f"motor_angle_rad={[round(v, 6) for v in values]}"
        )
    print(f"  knee_angle_rad={[round(float(v), 6) for v in args.knee_angles]}")
    print()

    if not args.yes:
        answer = input("Write config/four_bar_transmission.yaml? Type YES: ").strip()
        if answer != "YES":
            print("Calibration not written.")
            return 1

    path = ROOT / "config" / "four_bar_transmission.yaml"
    backup = update_four_bar_yaml(
        path=path,
        joints=joints,
        knee_angles=[float(value) for value in args.knee_angles],
        captures=captures,
        enable_after=bool(args.enable_after),
    )
    print(f"Wrote {path}")
    print(f"Backup: {backup}")
    print(f"four_bar_transmission.enabled={bool(args.enable_after)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
