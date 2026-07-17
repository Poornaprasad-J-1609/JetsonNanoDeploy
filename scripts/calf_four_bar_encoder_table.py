#!/usr/bin/env python3
"""One calf motor -> four-bar knee position, velocity and torque table.

This script does NOT enable the motor; move the unloaded linkage slowly by hand.

Calibration uses only the two known mechanical stops:
    maximum extension -> knee angle 0.00 rad
    maximum crouch    -> knee angle 1.56 rad
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


# Four-bar dimensions from the drawing, in mm.
AB = 40.00       # motor crank
BC = 343.47      # coupler
CD = 54.84       # knee rocker
DA = 339.73      # fixed thigh link

# Known calf/knee limits.
KNEE_AT_EXTENSION = 0.00
KNEE_AT_CROUCH = 1.56

# Default branch hint. The CLI defaults to trying both assembly branches.
DEFAULT_ASSEMBLY_BRANCH = -1
APPROX_CRANK_AT_EXTENSION = math.radians(-90.0)


def output_angle(theta, assembly_branch=DEFAULT_ASSEMBLY_BRANCH):
    """Four-bar forward kinematics: motor crank theta -> knee rocker phi."""
    bx, by = AB * math.cos(theta), AB * math.sin(theta)
    dx, dy = DA - bx, -by
    length_bd = math.hypot(dx, dy)

    if not abs(BC - CD) <= length_bd <= BC + CD:
        raise ValueError("four-bar cannot close at this motor angle")

    ex, ey = dx / length_bd, dy / length_bd
    along = (BC**2 - CD**2 + length_bd**2) / (2.0 * length_bd)
    height = math.sqrt(max(0.0, BC**2 - along**2))
    px, py = bx + along * ex, by + along * ey

    cx = px + float(assembly_branch) * height * (-ey)
    cy = py + float(assembly_branch) * height * ex
    return math.atan2(cy, cx - DA)


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def four_bar_jacobian(theta, assembly_branch=DEFAULT_ASSEMBLY_BRANCH):
    """Numerical Jacobian d(knee-rocker angle)/d(motor-crank angle)."""
    eps = 1e-6
    return angle_difference(
        output_angle(theta + eps, assembly_branch),
        output_angle(theta - eps, assembly_branch),
    ) / (2.0 * eps)


def reachable_output_travel_range(encoder_travel):
    """Estimate possible rocker travel ranges for the captured encoder travel."""
    ranges = []
    for branch in (-1.0, +1.0):
        for motor_direction in (-1.0, +1.0):
            crank_travel = motor_direction * float(encoder_travel)
            values = []
            for index in range(1440):
                theta = -math.pi + 2.0 * math.pi * index / 1440
                try:
                    values.append(
                        angle_difference(
                            output_angle(theta + crank_travel, branch),
                            output_angle(theta, branch),
                        )
                    )
                except ValueError:
                    pass
            if values:
                ranges.append(
                    {
                        "assembly_branch": branch,
                        "motor_direction": motor_direction,
                        "min": min(values),
                        "max": max(values),
                    }
                )
    return ranges


def fit_endpoint_calibration(
    encoder_extension,
    encoder_crouch,
    assembly_branch="auto",
    knee_at_crouch=KNEE_AT_CROUCH,
):
    """Fit the unknown absolute crank phase using only the two end stops."""
    encoder_travel = encoder_crouch - encoder_extension
    knee_travel = float(knee_at_crouch) - KNEE_AT_EXTENSION
    if abs(encoder_travel) < 0.05:
        raise ValueError("extension and crouch encoder readings are too close")

    candidates = []
    grid_count = 1440
    if str(assembly_branch).lower() == "auto":
        assembly_branches = (-1.0, +1.0)
    else:
        assembly_branches = (float(assembly_branch),)

    for branch in assembly_branches:
        if branch not in (-1.0, +1.0):
            raise ValueError("--assembly-branch must be -1, 1, or auto")
        for motor_direction in (-1.0, +1.0):
            crank_travel = motor_direction * encoder_travel
            for output_sign in (-1.0, +1.0):
                target_output_travel = output_sign * knee_travel

                def error(theta_extension):
                    theta_crouch = theta_extension + crank_travel
                    actual = angle_difference(
                        output_angle(theta_crouch, branch),
                        output_angle(theta_extension, branch),
                    )
                    return actual - target_output_travel

                previous_theta = -math.pi
                previous_error = error(previous_theta)
                for index in range(1, grid_count + 1):
                    theta = -math.pi + 2.0 * math.pi * index / grid_count
                    current_error = error(theta)

                    if previous_error * current_error <= 0.0:
                        lo, hi = previous_theta, theta
                        flo = previous_error
                        for _ in range(50):
                            mid = 0.5 * (lo + hi)
                            fmid = error(mid)
                            if flo * fmid <= 0.0:
                                hi = mid
                            else:
                                lo, flo = mid, fmid
                        theta_extension = 0.5 * (lo + hi)

                        if abs(error(theta_extension)) < 1e-5:
                            knee_direction = output_sign
                            monotonic = True
                            for sample in range(41):
                                fraction = sample / 40.0
                                theta_sample = theta_extension + fraction * crank_travel
                                total_j = (
                                    knee_direction
                                    * four_bar_jacobian(theta_sample, branch)
                                    * motor_direction
                                )
                                if total_j * encoder_travel <= 0.0 or abs(total_j) < 1e-4:
                                    monotonic = False
                                    break

                            if monotonic:
                                score = abs(
                                    angle_difference(
                                        theta_extension,
                                        APPROX_CRANK_AT_EXTENSION,
                                    )
                                )
                                candidates.append(
                                    (
                                        score,
                                        theta_extension,
                                        motor_direction,
                                        knee_direction,
                                        branch,
                                    )
                                )

                    previous_theta, previous_error = theta, current_error

    if not candidates:
        diagnostics = reachable_output_travel_range(encoder_travel)
        if diagnostics:
            max_abs = max(
                max(abs(item["min"]), abs(item["max"])) for item in diagnostics
            )
            detail = (
                f" Captured encoder travel is {encoder_travel:+.6f} rad; "
                f"with the current link dimensions the largest reachable "
                f"rocker travel for that motor travel is about {max_abs:.3f} rad, "
                f"but --knee-at-crouch asks for {knee_travel:.3f} rad."
            )
        else:
            detail = " No valid closure range was found for the captured travel."
        raise ValueError(
            "the two encoder limits are inconsistent with the four-bar dimensions; "
            "check the endpoints, --knee-at-crouch, link dimensions, or --assembly-branch."
            + detail
        )

    _, theta_extension, motor_direction, knee_direction, branch = min(candidates)
    return {
        "encoder_extension": encoder_extension,
        "encoder_crouch": encoder_crouch,
        "theta_extension": theta_extension,
        "motor_direction": motor_direction,
        "knee_direction": knee_direction,
        "assembly_branch": branch,
        "phi_extension": output_angle(theta_extension, branch),
    }


def motor_to_knee(
    motor_position,
    motor_velocity,
    motor_torque,
    calibration,
    clamp_knee=False,
    knee_at_crouch=KNEE_AT_CROUCH,
    clamp_tolerance=0.01,
):
    """Return knee position, velocity, and torque from motor feedback."""
    assembly_branch = float(calibration["assembly_branch"])
    theta = calibration["theta_extension"] + calibration["motor_direction"] * (
        motor_position - calibration["encoder_extension"]
    )
    phi = output_angle(theta, assembly_branch)

    jacobian = (
        calibration["knee_direction"]
        * four_bar_jacobian(theta, assembly_branch)
        * calibration["motor_direction"]
    )
    if abs(jacobian) < 1e-4:
        raise ZeroDivisionError("four-bar is too close to a singularity")

    knee_position = KNEE_AT_EXTENSION + calibration["knee_direction"] * angle_difference(
        phi, calibration["phi_extension"]
    )
    if clamp_knee:
        knee_min = float(KNEE_AT_EXTENSION)
        knee_max = float(knee_at_crouch)
        tolerance = max(0.0, float(clamp_tolerance))
        if knee_min - tolerance <= knee_position < knee_min:
            knee_position = knee_min
        elif knee_max < knee_position <= knee_max + tolerance:
            knee_position = knee_max
    knee_velocity = jacobian * motor_velocity
    knee_torque = motor_torque / jacobian
    return knee_position, knee_velocity, knee_torque


def format_table_header(columns):
    return " | ".join(f"{name:>20}" for name in columns)


def format_table_values(values):
    return " | ".join(f"{value:+20.6f}" for value in values)


def print_live_values(values):
    print("\r\033[2K" + format_table_values(values), end="", flush=True)


def profile_name_for_joint(joint_name):
    return str(joint_name).replace("_joint", "")


def virtual_sign_for_joint(joint_name):
    return -1 if str(joint_name).startswith(("FR_", "BR_")) else 1


def generated_lookup_table(
    joint_name,
    layer,
    encoder_extension,
    encoder_crouch,
    calibration,
    points,
):
    points = int(points)
    if points < 3:
        raise ValueError("lookup table needs at least 3 points")
    offset = float(layer.joint_offsets[joint_name])
    direction = float(layer.joint_directions[joint_name])
    motor_angles = []
    knee_angles = []
    for index in range(points):
        ratio = index / float(points - 1)
        raw_encoder = float(encoder_extension) + ratio * (
            float(encoder_crouch) - float(encoder_extension)
        )
        motor_theta = direction * (raw_encoder - offset)
        knee_angle, _, _ = motor_to_knee(raw_encoder, 0.0, 0.0, calibration)
        motor_angles.append(round(float(motor_theta), 6))
        knee_angles.append(round(float(knee_angle), 6))
    return motor_angles, knee_angles


def trim_low_jacobian_edges(motor_angles, knee_angles, min_abs_jacobian):
    """Drop only edge samples that are too close to a four-bar toggle."""
    motor_angles = list(motor_angles)
    knee_angles = list(knee_angles)
    threshold = float(min_abs_jacobian)
    trimmed_front = 0
    trimmed_back = 0

    while len(motor_angles) > 3:
        slopes = [
            abs((knee_angles[i + 1] - knee_angles[i]) / (motor_angles[i + 1] - motor_angles[i]))
            for i in range(len(motor_angles) - 1)
        ]
        low_indices = [i for i, slope in enumerate(slopes) if slope < threshold]
        if not low_indices:
            break

        if low_indices[0] == 0:
            motor_angles.pop(0)
            knee_angles.pop(0)
            trimmed_front += 1
        elif low_indices[-1] == len(slopes) - 1:
            motor_angles.pop()
            knee_angles.pop()
            trimmed_back += 1
        else:
            break

    return motor_angles, knee_angles, trimmed_front, trimmed_back


def finite_monotonic(values):
    if len(values) < 3:
        return False
    if not all(math.isfinite(float(value)) for value in values):
        return False
    deltas = [b - a for a, b in zip(values, values[1:])]
    return all(delta > 0.0 for delta in deltas) or all(delta < 0.0 for delta in deltas)


def profile_is_valid(profile):
    motor = [float(v) for v in profile.get("motor_angle_rad", [])]
    knee = [float(v) for v in profile.get("knee_angle_rad", [])]
    return len(motor) == len(knee) and finite_monotonic(motor) and finite_monotonic(knee)


def write_four_bar_yaml(
    joint_name,
    layer,
    encoder_extension,
    encoder_crouch,
    calibration,
    table_points,
    enable_yaml,
):
    path = ROOT / "config" / "four_bar_transmission.yaml"
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    root = cfg.setdefault("four_bar_transmission", {})
    profiles = root.setdefault("profiles", {})
    joints = root.setdefault("joints", {})
    profile_name = profile_name_for_joint(joint_name)
    profile = profiles.setdefault(profile_name, {})
    min_abs_jacobian = float(profile.get("min_abs_jacobian", 0.05))
    motor_angles, knee_angles = generated_lookup_table(
        joint_name=joint_name,
        layer=layer,
        encoder_extension=encoder_extension,
        encoder_crouch=encoder_crouch,
        calibration=calibration,
        points=table_points,
    )
    motor_angles, knee_angles, trimmed_front, trimmed_back = trim_low_jacobian_edges(
        motor_angles,
        knee_angles,
        min_abs_jacobian=min_abs_jacobian,
    )
    if trimmed_front or trimmed_back:
        print(
            "Trimmed four-bar lookup near toggle: "
            f"front={trimmed_front}, back={trimmed_back}, "
            f"min_abs_jacobian={min_abs_jacobian:.4f}"
        )

    profile["motor_angle_rad"] = motor_angles
    profile["knee_angle_rad"] = knee_angles
    profile.setdefault("efficiency", 1.0)
    profile.setdefault("motor_torque_limit_nm", 120.0)
    profile.setdefault("min_abs_jacobian", 0.05)
    profile.setdefault("endpoint_tolerance_rad", 0.01)
    profile.setdefault("clamp_outside_calibration", False)
    profile.setdefault("compensate_efficiency_in_commands", True)

    joints[joint_name] = {
        "enabled": True,
        "profile": profile_name,
        "virtual_sign": virtual_sign_for_joint(joint_name),
    }

    required_calf_joints = (
        "FR_calf_joint",
        "FL_calf_joint",
        "BR_calf_joint",
        "BL_calf_joint",
    )
    for other_joint, item in list(joints.items()):
        other_profile = profiles.get(str((item or {}).get("profile", "")), {})
        if not profile_is_valid(other_profile):
            item["enabled"] = False

    calibrated_calf_joints = []
    for calf_joint in required_calf_joints:
        item = joints.get(calf_joint, {}) or {}
        other_profile = profiles.get(str(item.get("profile", "")), {})
        if bool(item.get("enabled", False)) and profile_is_valid(other_profile):
            calibrated_calf_joints.append(calf_joint)

    if enable_yaml:
        missing = [
            name for name in required_calf_joints if name not in calibrated_calf_joints
        ]
        if missing:
            root["enabled"] = False
            print(
                "WARNING: four-bar remains globally disabled until all calf "
                "profiles are calibrated: " + ", ".join(missing)
            )
        else:
            root["enabled"] = True
            print("Four-bar transmission globally enabled for all calf joints.")
    else:
        root["enabled"] = False
        print(
            "Four-bar lookup was written, but global four-bar is still disabled. "
            "Use --enable-yaml after all four calf profiles are calibrated."
        )

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    tmp.replace(path)
    return path, backup, motor_angles, knee_angles


def capture_encoder(
    prompt,
    buses,
    layer,
    poll_commands,
    active_key,
    joint_name,
    motor_ids,
    joint_can_bus,
    feedback_by_bus_motor_id,
    feedback_by_motor_id,
):
    """Wait for Enter, then average fresh encoder feedback at that stop."""
    input(prompt)
    readings = []
    last_timestamp = None
    deadline = time.monotonic() + 2.0
    while len(readings) < 20 and time.monotonic() < deadline:
        layer.send_raw_commands(buses, poll_commands)
        frames = MotorCommandLayer.read_all_frames(buses, timeout=0.05)
        update_feedback_from_frames(
            frames,
            layer,
            active_key,
            feedback_by_bus_motor_id,
            feedback_by_motor_id,
        )
        feedback = feedback_for_joint(
            joint_name,
            motor_ids,
            joint_can_bus,
            feedback_by_bus_motor_id,
            feedback_by_motor_id,
        )
        timestamp = None if feedback is None else feedback.get("timestamp")
        if (
            feedback is not None
            and int(feedback.get("fault_bits", 0)) == 0
            and timestamp != last_timestamp
        ):
            readings.append(float(feedback["position"]))
            last_timestamp = timestamp
        time.sleep(0.01)

    if not readings:
        raise RuntimeError("no valid motor encoder feedback was received")
    encoder = sum(readings) / len(readings)
    print(f"Captured encoder: {encoder:+.6f} rad")
    return encoder


def main():
    parser = argparse.ArgumentParser()
    add_can_topology_args(parser, default_port="/dev/ttyUSB0", default_can_count=2)
    parser.add_argument(
        "--calf-joint",
        default="FR_calf_joint",
        choices=("FR_calf_joint", "FL_calf_joint", "BR_calf_joint", "BL_calf_joint"),
    )
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl+C")
    parser.add_argument("--csv", default="calf_four_bar_table.csv")
    parser.add_argument("--extension-encoder", type=float, default=None)
    parser.add_argument("--crouch-encoder", type=float, default=None)
    parser.add_argument(
        "--assembly-branch",
        choices=("auto", "-1", "1"),
        default="auto",
        help="four-bar circle-intersection branch; auto tries both",
    )
    parser.add_argument(
        "--knee-at-crouch",
        type=float,
        default=KNEE_AT_CROUCH,
        help="known knee angle at maximum crouch in radians",
    )
    parser.add_argument(
        "--scroll",
        action="store_true",
        help="print every sample on a new line instead of updating one live row",
    )
    parser.add_argument(
        "--write-yaml",
        action="store_true",
        help="write the generated lookup table into config/four_bar_transmission.yaml",
    )
    parser.add_argument(
        "--enable-yaml",
        action="store_true",
        help="also set four_bar_transmission.enabled=true after writing YAML",
    )
    parser.add_argument(
        "--table-points",
        type=int,
        default=25,
        help="number of generated lookup samples between extension and crouch",
    )
    args = parser.parse_args()
    if (args.extension_encoder is None) != (args.crouch_encoder is None):
        parser.error("provide both --extension-encoder and --crouch-encoder")

    joint_cfg, motor_cfg = load_config()
    policy_order = list(joint_cfg["policy_to_real_order"])
    motor_ids = motor_cfg["motor_ids"]
    joint_can_bus = resolve_joint_can_bus(policy_order, args.can_count)
    port_by_bus = resolve_port_by_bus(args)

    layer = MotorCommandLayer(
        policy_order=policy_order,
        motor_ids=motor_ids,
        active_joints=[args.calf_joint],
        joint_can_bus=joint_can_bus,
    )
    active_ports = ports_for_active_joints(
        port_by_bus, joint_can_bus, layer.active_joints
    )
    buses = open_can_buses(
        active_ports,
        baud=args.baud,
        timeout=0.002,
        backend=args.can_backend,
        bitrate=args.can_bitrate,
    )

    poll_commands = layer.build_feedback_poll_commands()
    stop_commands = layer.build_stop_commands()
    layer.send_raw_commands(buses, stop_commands)

    feedback_by_bus_motor_id = {}
    feedback_by_motor_id = {}
    motor_id = int(motor_ids[args.calf_joint])
    bus_name = joint_can_bus.get(args.calf_joint, "front")
    active_key = {(bus_name, motor_id)}
    dt = 1.0 / args.rate

    try:
        if args.extension_encoder is None:
            print("\nTWO-STOP CALIBRATION (motor remains disabled)")
            encoder_extension = capture_encoder(
                "Move the leg to MAXIMUM EXTENSION, hold it, then press Enter: ",
                buses, layer, poll_commands, active_key, args.calf_joint,
                motor_ids, joint_can_bus, feedback_by_bus_motor_id, feedback_by_motor_id,
            )
            encoder_crouch = capture_encoder(
                "Move the leg to MAXIMUM CROUCH, hold it, then press Enter: ",
                buses, layer, poll_commands, active_key, args.calf_joint,
                motor_ids, joint_can_bus, feedback_by_bus_motor_id, feedback_by_motor_id,
            )
        else:
            encoder_extension = args.extension_encoder
            encoder_crouch = args.crouch_encoder

        calibration = fit_endpoint_calibration(
            encoder_extension,
            encoder_crouch,
            assembly_branch=args.assembly_branch,
            knee_at_crouch=args.knee_at_crouch,
        )
    except BaseException:
        layer.send_raw_commands(buses, stop_commands)
        close_can_buses(buses)
        raise

    print(
        "Calibration complete: "
        f"extension={encoder_extension:+.6f}, crouch={encoder_crouch:+.6f} rad, "
        f"assembly_branch={calibration['assembly_branch']:+.0f}, "
        f"motor_direction={calibration['motor_direction']:+.0f}, "
        f"knee_direction={calibration['knee_direction']:+.0f}"
    )
    if args.write_yaml:
        yaml_path, backup_path, motor_table, knee_table = write_four_bar_yaml(
            joint_name=args.calf_joint,
            layer=layer,
            encoder_extension=encoder_extension,
            encoder_crouch=encoder_crouch,
            calibration=calibration,
            table_points=args.table_points,
            enable_yaml=args.enable_yaml,
        )
        print(f"Wrote four-bar lookup for {args.calf_joint}: {yaml_path}")
        print(f"Backup: {backup_path}")
        print(f"motor_angle_rad: {motor_table}")
        print(f"knee_angle_rad:  {knee_table}")
    end_time = None if args.seconds <= 0 else time.monotonic() + args.seconds

    columns = [
        "motor_position_rad",
        "motor_velocity_rad_s",
        "motor_torque_Nm",
        "knee_position_rad",
        "knee_velocity_rad_s",
        "knee_torque_Nm",
    ]
    csv_path = Path(args.csv).expanduser().resolve()

    print(f"Reading only {args.calf_joint}; motor remains disabled.")
    print("Move the unloaded linkage slowly by hand. Press Ctrl+C to stop.")
    print(format_table_header(columns))
    print("-" * 137)
    live_display = sys.stdout.isatty() and not args.scroll
    live_row_started = False
    live_row_closed = False

    try:
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()

            while end_time is None or time.monotonic() < end_time:
                start = time.monotonic()
                layer.send_raw_commands(buses, poll_commands)
                frames = MotorCommandLayer.read_all_frames(
                    buses, timeout=min(0.05, dt)
                )
                update_feedback_from_frames(
                    frames,
                    layer,
                    active_key,
                    feedback_by_bus_motor_id,
                    feedback_by_motor_id,
                )
                feedback = feedback_for_joint(
                    args.calf_joint,
                    motor_ids,
                    joint_can_bus,
                    feedback_by_bus_motor_id,
                    feedback_by_motor_id,
                )

                if feedback is not None and int(feedback.get("fault_bits", 0)) == 0:
                    motor_values = (
                        float(feedback["position"]),
                        float(feedback["velocity"]),
                        float(feedback["torque"]),
                    )
                    knee_values = motor_to_knee(
                        *motor_values,
                        calibration,
                        clamp_knee=True,
                        knee_at_crouch=args.knee_at_crouch,
                    )
                    values = motor_values + knee_values
                    row = dict(zip(columns, values))
                    writer.writerow(row)
                    if live_display:
                        print_live_values(values)
                        live_row_started = True
                    else:
                        print(format_table_values(values))

                time.sleep(max(0.0, dt - (time.monotonic() - start)))

    except KeyboardInterrupt:
        if live_row_started:
            print()
            live_row_closed = True
        print("\nStopped.")
    finally:
        if live_row_started and not live_row_closed:
            print()
        layer.send_raw_commands(buses, stop_commands)
        close_can_buses(buses)

    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
