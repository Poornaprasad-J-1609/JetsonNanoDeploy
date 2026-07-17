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

# This selects the physical assembly branch. Change to +1 if the solver says
# the endpoints are inconsistent with the linkage dimensions.
ASSEMBLY_BRANCH = -1
APPROX_CRANK_AT_EXTENSION = math.radians(-90.0)


def output_angle(theta):
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

    cx = px + ASSEMBLY_BRANCH * height * (-ey)
    cy = py + ASSEMBLY_BRANCH * height * ex
    return math.atan2(cy, cx - DA)


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def four_bar_jacobian(theta):
    """Numerical Jacobian d(knee-rocker angle)/d(motor-crank angle)."""
    eps = 1e-6
    return angle_difference(
        output_angle(theta + eps), output_angle(theta - eps)
    ) / (2.0 * eps)


def fit_endpoint_calibration(encoder_extension, encoder_crouch):
    """Fit the unknown absolute crank phase using only the two end stops."""
    encoder_travel = encoder_crouch - encoder_extension
    knee_travel = KNEE_AT_CROUCH - KNEE_AT_EXTENSION
    if abs(encoder_travel) < 0.05:
        raise ValueError("extension and crouch encoder readings are too close")

    candidates = []
    grid_count = 1440

    for motor_direction in (-1.0, +1.0):
        crank_travel = motor_direction * encoder_travel
        for output_sign in (-1.0, +1.0):
            target_output_travel = output_sign * knee_travel

            def error(theta_extension):
                theta_crouch = theta_extension + crank_travel
                actual = angle_difference(
                    output_angle(theta_crouch), output_angle(theta_extension)
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
                                * four_bar_jacobian(theta_sample)
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
                                (score, theta_extension, motor_direction, knee_direction)
                            )

                previous_theta, previous_error = theta, current_error

    if not candidates:
        raise ValueError(
            "the two encoder limits are inconsistent with the four-bar dimensions; "
            "check the endpoints or change ASSEMBLY_BRANCH"
        )

    _, theta_extension, motor_direction, knee_direction = min(candidates)
    return {
        "encoder_extension": encoder_extension,
        "encoder_crouch": encoder_crouch,
        "theta_extension": theta_extension,
        "motor_direction": motor_direction,
        "knee_direction": knee_direction,
        "phi_extension": output_angle(theta_extension),
    }


def motor_to_knee(motor_position, motor_velocity, motor_torque, calibration):
    """Return knee position, velocity, and torque from motor feedback."""
    theta = calibration["theta_extension"] + calibration["motor_direction"] * (
        motor_position - calibration["encoder_extension"]
    )
    phi = output_angle(theta)

    jacobian = (
        calibration["knee_direction"]
        * four_bar_jacobian(theta)
        * calibration["motor_direction"]
    )
    if abs(jacobian) < 1e-4:
        raise ZeroDivisionError("four-bar is too close to a singularity")

    knee_position = KNEE_AT_EXTENSION + calibration["knee_direction"] * angle_difference(
        phi, calibration["phi_extension"]
    )
    knee_velocity = jacobian * motor_velocity
    knee_torque = motor_torque / jacobian
    return knee_position, knee_velocity, knee_torque


def format_table_header(columns):
    return " | ".join(f"{name:>20}" for name in columns)


def format_table_values(values):
    return " | ".join(f"{value:+20.6f}" for value in values)


def print_live_values(values):
    print("\r\033[2K" + format_table_values(values), end="", flush=True)


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
        "--scroll",
        action="store_true",
        help="print every sample on a new line instead of updating one live row",
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

        calibration = fit_endpoint_calibration(encoder_extension, encoder_crouch)
    except BaseException:
        layer.send_raw_commands(buses, stop_commands)
        close_can_buses(buses)
        raise

    print(
        "Calibration complete: "
        f"extension={encoder_extension:+.6f}, crouch={encoder_crouch:+.6f} rad"
    )
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
                    knee_values = motor_to_knee(*motor_values, calibration)
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
