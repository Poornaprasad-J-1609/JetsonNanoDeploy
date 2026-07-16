#!/usr/bin/env python3
"""Install the nonlinear four-bar adapter into JetsonNanoDeploy.

Expected target: branch walk-branch-jun18.
The script is idempotent and creates .fourbar_backup files before edits.
The installed YAML remains disabled until measured calibration is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import textwrap

BUNDLE_ROOT = Path(__file__).resolve().parent
PAYLOAD_ROOT = BUNDLE_ROOT / "payload"


STATE_REPLACEMENT = r'''offset = float(self.motor_layer.joint_offsets[joint_name])
direction = float(self.motor_layer.joint_directions[joint_name])
position_raw = float(feedback["position"])
velocity_raw = float(feedback["velocity"])
torque_raw = float(feedback["torque"])

if hasattr(self.motor_layer, "decode_joint_feedback"):
    mapped = self.motor_layer.decode_joint_feedback(
        joint_name=joint_name,
        position_raw=position_raw,
        velocity_raw=velocity_raw,
        torque_raw=torque_raw,
    )
    q_joint = float(mapped["joint_position"])
    qd_joint = float(mapped["joint_velocity"])
    tau_joint = float(mapped["joint_torque"])
    motor_position = float(mapped["motor_position"])
    motor_velocity = float(mapped["motor_velocity"])
    motor_torque = float(mapped["motor_torque"])
    transmission_jacobian = float(
        mapped["transmission_jacobian"]
    )
    transmission_efficiency = float(
        mapped["transmission_efficiency"]
    )
    transmission_enabled = bool(
        mapped["transmission_enabled"]
    )
else:
    q_joint = motor_position_to_joint_angle(
        position_raw,
        offset=offset,
        direction=direction,
    )
    qd_joint = direction * velocity_raw
    tau_joint = direction * torque_raw
    motor_position = q_joint
    motor_velocity = qd_joint
    motor_torque = tau_joint
    transmission_jacobian = 1.0
    transmission_efficiency = 1.0
    transmission_enabled = False

self.q_current[index] = q_joint
self.qd_current[index] = qd_joint
feedback = dict(feedback)
feedback["timestamp"] = timestamp
feedback["bus_name"] = bus_name
feedback["position_raw"] = position_raw
feedback["velocity_raw"] = velocity_raw
feedback["torque_raw"] = torque_raw
feedback["motor_position"] = motor_position
feedback["motor_velocity"] = motor_velocity
feedback["motor_torque"] = motor_torque
feedback["joint_position"] = q_joint
feedback["joint_velocity"] = qd_joint
feedback["joint_torque"] = tau_joint
feedback["position"] = q_joint
feedback["velocity"] = qd_joint
feedback["torque"] = tau_joint
feedback["joint_direction"] = direction
feedback["transmission_jacobian"] = transmission_jacobian
feedback["transmission_efficiency"] = transmission_efficiency
feedback["transmission_enabled"] = transmission_enabled'''


def backup(path: Path) -> None:
    backup_path = path.with_name(path.name + ".fourbar_backup")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def copy_payload(repo: Path) -> None:
    relative_paths = (
        "src/four_bar_transmission.py",
        "src/four_bar_motor_command_layer.py",
        "config/four_bar_transmission.yaml",
        "tests/test_four_bar_transmission.py",
        "docs/FOUR_BAR_DEPLOYMENT.md",
    )
    for relative in relative_paths:
        source = PAYLOAD_ROOT / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup(destination)
        shutil.copy2(source, destination)
        print(f"installed {relative}")


def patch_main_controller(repo: Path) -> None:
    path = repo / "src" / "main_controller.py"
    text = path.read_text(encoding="utf-8")
    if "from four_bar_motor_command_layer import" in text:
        print("src/main_controller.py already patched")
        return

    old = "from motor_command_layer import MotorCommandLayer, print_mit_commands"
    new = (
        "from four_bar_motor_command_layer import "
        "FourBarMotorCommandLayer as MotorCommandLayer\n"
        "from motor_command_layer import print_mit_commands"
    )
    if old not in text:
        raise RuntimeError(
            "Could not find the expected MotorCommandLayer import in "
            "src/main_controller.py. The branch may have changed."
        )
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("patched src/main_controller.py")


def patch_state_estimator(repo: Path) -> None:
    path = repo / "src" / "state_estimator.py"
    text = path.read_text(encoding="utf-8")
    if 'mapped = self.motor_layer.decode_joint_feedback(' in text:
        print("src/state_estimator.py already patched")
        return

    start_token = "offset = float(self.motor_layer.joint_offsets[joint_name])"
    end_token = 'feedback["joint_direction"] = direction'
    token_start = text.find(start_token)
    if token_start < 0:
        raise RuntimeError(
            "Could not find the current feedback conversion block start in "
            "src/state_estimator.py. The branch may have changed."
        )
    start = text.rfind("\n", 0, token_start) + 1
    indent = text[start:token_start]
    if indent.strip():
        raise RuntimeError("Unexpected indentation before feedback block")

    token_end = text.find(end_token, token_start)
    if token_end < 0:
        raise RuntimeError(
            "Could not find the current feedback conversion block end in "
            "src/state_estimator.py. The branch may have changed."
        )
    end = token_end + len(end_token)
    replacement = textwrap.indent(STATE_REPLACEMENT, indent)

    backup(path)
    patched = text[:start] + replacement + text[end:]
    path.write_text(patched, encoding="utf-8")
    print("patched src/state_estimator.py")


def validate_target(repo: Path) -> None:
    required = (
        repo / "src" / "main_controller.py",
        repo / "src" / "state_estimator.py",
        repo / "src" / "motor_command_layer.py",
        repo / "config" / "joint_limits.yaml",
        repo / "config" / "joint_offsets.yaml",
        repo / "config" / "motor_directions.yaml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Not a compatible JetsonNanoDeploy checkout. Missing:\n" + lines
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=".",
        help="JetsonNanoDeploy repository root (default: current directory)",
    )
    args = parser.parse_args()
    repo = Path(args.repo).expanduser().resolve()

    try:
        validate_target(repo)
        copy_payload(repo)
        patch_main_controller(repo)
        patch_state_estimator(repo)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\nFour-bar adapter installed but intentionally DISABLED.")
    print("Next steps:")
    print("  1. Fill config/four_bar_transmission.yaml with measured samples.")
    print("  2. Cover every calf hard limit, or reduce joint_limits/poses.")
    print(
        "  3. Run: PYTHONPATH=src python3 -m unittest "
        "tests/test_four_bar_transmission.py"
    )
    print("  4. Set enabled: true only after suspended-leg validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
