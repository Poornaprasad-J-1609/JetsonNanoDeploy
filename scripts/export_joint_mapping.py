#!/usr/bin/env python3
"""Export the current name-based policy-to-motor candidate table."""

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from joint_mapping import POLICY_JOINT_ORDER  # noqa: E402


def load_yaml(name):
    with (ROOT / "config" / name).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    motor_ids = load_yaml("motor_ids.yaml")["motor_ids"]
    directions = load_yaml("motor_directions.yaml")["motor_directions"]
    offsets = load_yaml("joint_offsets.yaml")["joint_offsets"]
    limits = load_yaml("joint_limits.yaml")["joint_limits"]
    defaults = load_yaml("default_pose.yaml")["default_pose"]
    four_bar = load_yaml("four_bar_transmission.yaml")["four_bar_transmission"]
    four_bar_enabled = bool(four_bar.get("enabled", False))
    four_bar_joints = four_bar.get("joints", {}) or {}

    rows = []
    for index, name in enumerate(POLICY_JOINT_ORDER):
        transmission = four_bar_joints.get(name, {}) or {}
        rows.append(
            {
                "policy_index": index,
                "policy_joint": name,
                "motor_id": int(motor_ids[name]),
                "encoder_sign": int(directions[name]),
                "target_sign": int(directions[name]),
                "encoder_offset_rad": float(offsets[name]),
                "four_bar_global_enabled": four_bar_enabled,
                "virtual_joint_conversion": (
                    str(transmission.get("profile"))
                    if bool(transmission.get("enabled", False))
                    else "identity"
                ),
                "minimum_rad": float(limits[name]["min"]),
                "maximum_rad": float(limits[name]["max"]),
                "q_default_rad": float(defaults[name]),
                "verification": "UNVERIFIED_FROM_MODEL_12357_TRAINING_SOURCE",
            }
        )

    fields = list(rows[0])
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote:", output)
    else:
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
