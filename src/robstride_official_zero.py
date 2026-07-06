#!/usr/bin/env python3
"""Read and set persistent motor zero using RobStride's official SDK.

This utility intentionally uses raw motor coordinates. It does not apply the
deployment direction map, modulo wrapping, pose snapping, or software offsets.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import yaml

try:
    from robstride_dynamics import (
        CommunicationType,
        Motor,
        ParameterType,
        RobstrideBus,
    )
except ImportError as exc:
    raise ImportError(
        "Install the official SDK first: "
        "/usr/bin/python3 -m pip install robstride-dynamics"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DISPLAY_LEGS = ("FR", "FL", "BR", "BL")
PARAMETERS = {
    "mechanical_position": ParameterType.MECHANICAL_POSITION,
    "measured_position": ParameterType.MEASURED_POSITION,
    "mechanical_offset": ParameterType.MECHANICAL_OFFSET,
    "zero_state": ParameterType.ZERO_STATE,
}


class TimeoutRobstrideBus(RobstrideBus):
    """Official bus with a finite default receive timeout."""

    def receive(self, timeout=None):
        return super().receive(timeout=0.25 if timeout is None else timeout)

    def receive_read_frame(self):
        response = self.receive(timeout=0.25)
        if response is None:
            raise TimeoutError("no parameter response from motor")
        communication_type, _, _, data = response
        if communication_type != CommunicationType.READ_PARAMETER:
            raise RuntimeError(
                f"expected parameter response type "
                f"{CommunicationType.READ_PARAMETER}, got {communication_type}"
            )
        if len(data) < 8:
            raise RuntimeError(f"short parameter response: {len(data)} byte(s)")
        return data[4:]


def load_motor_ids():
    with open(ROOT / "config" / "motor_ids.yaml", "r") as stream:
        return yaml.safe_load(stream)["motor_ids"]


def display_order(motor_ids):
    ordered = []
    for leg in DISPLAY_LEGS:
        names = [name for name in motor_ids if name.startswith(leg + "_")]
        ordered.extend(sorted(names, key=lambda name: int(motor_ids[name])))
    remaining = [name for name in motor_ids if name not in ordered]
    ordered.extend(sorted(remaining, key=lambda name: int(motor_ids[name])))
    return ordered


def logical_bus_for_joint(joint_name, can_count):
    if can_count == 1:
        return "can0"
    if can_count == 2:
        return "front" if joint_name.startswith(("FR_", "FL_")) else "back"
    return joint_name.split("_", 1)[0]


def channel_map(can_count, channels):
    names = {
        1: ("can0",),
        2: ("front", "back"),
        4: ("FR", "FL", "BR", "BL"),
    }[can_count]
    if len(channels) != len(names):
        raise ValueError(
            f"--can-count {can_count} requires {len(names)} channel(s) "
            f"in this order: {' '.join(names)}"
        )
    return dict(zip(names, channels))


def build_buses(motor_ids, can_count, channels, model, bitrate):
    channels_by_bus = channel_map(can_count, channels)
    motors_by_bus = {name: {} for name in channels_by_bus}
    joint_bus = {}
    for joint_name, motor_id in motor_ids.items():
        bus_name = logical_bus_for_joint(joint_name, can_count)
        joint_bus[joint_name] = bus_name
        motors_by_bus[bus_name][joint_name] = Motor(id=int(motor_id), model=model)

    buses = {
        bus_name: TimeoutRobstrideBus(
            channel=channels_by_bus[bus_name],
            motors=motors,
            bitrate=bitrate,
        )
        for bus_name, motors in motors_by_bus.items()
    }
    return buses, joint_bus, channels_by_bus


def safe_read(bus, joint_name, parameter):
    try:
        value = bus.read(joint_name, parameter)
        value = float(value)
        return value if np.isfinite(value) else None, "non-finite response"
    except Exception as exc:
        return None, str(exc)


def read_snapshot(buses, joint_bus, motor_ids, channels_by_bus, phase):
    rows = []
    for joint_name in display_order(motor_ids):
        bus_name = joint_bus[joint_name]
        bus = buses[bus_name]
        values = {}
        errors = []
        for field_name, parameter in PARAMETERS.items():
            value, error = safe_read(bus, joint_name, parameter)
            values[field_name] = value
            if error:
                errors.append(f"{field_name}: {error}")
        rows.append({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "phase": phase,
            "joint_name": joint_name,
            "logical_bus": bus_name,
            "channel": channels_by_bus[bus_name],
            "motor_id": int(motor_ids[joint_name]),
            **values,
            "error": " | ".join(errors),
        })
    return rows


def print_snapshot(rows):
    print("-" * 132)
    print(
        f"{'Joint':20s} | {'Channel':8s} | {'ID':4s} | {'State':10s} | "
        f"{'mechPos':>11s} | {'measPos':>11s} | {'mechOffset':>11s} | {'zero':>5s}"
    )
    print("-" * 132)
    connected = 0
    for row in rows:
        position = row["mechanical_position"]
        state = "CONNECTED" if position is not None else "NO_REPLY"
        connected += int(position is not None)

        def value_text(value, width=11):
            return f"{value:+{width}.6f}" if value is not None else f"{'-':>{width}s}"

        zero_state = row["zero_state"]
        zero_text = "-" if zero_state is None else str(int(round(zero_state)))
        print(
            f"{row['joint_name']:20s} | {row['channel']:8s} | "
            f"0x{row['motor_id']:02X} | {state:10s} | "
            f"{value_text(position)} | "
            f"{value_text(row['measured_position'])} | "
            f"{value_text(row['mechanical_offset'])} | {zero_text:>5s}"
        )
        if row["error"]:
            print(f"  parameter warning: {row['error']}")
    print("-" * 132)
    print(f"Connected mechanical-position reads: {connected}/{len(rows)}")
    return connected


def append_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "phase",
        "joint_name",
        "logical_bus",
        "channel",
        "motor_id",
        "mechanical_position",
        "measured_position",
        "mechanical_offset",
        "zero_state",
        "error",
    ]
    new_file = not path.exists()
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def receive_optional_status(bus, joint_name, timeout=0.25):
    response = bus.receive(timeout=timeout)
    if response is None:
        return False, "no acknowledgement"
    communication_type, _, _, _ = response
    valid = communication_type in (
        CommunicationType.OPERATION_STATUS,
        CommunicationType.SAVE_PARAMETERS,
    )
    return valid, f"response type={communication_type}"


def set_zero_all(buses, joint_bus, motor_ids, save_parameters):
    failures = []
    completed = []
    for joint_name in display_order(motor_ids):
        bus = buses[joint_bus[joint_name]]
        motor_id = int(motor_ids[joint_name])
        try:
            bus.disable(joint_name)
            data = bytes([1]) + bytes(7)
            bus.transmit(
                CommunicationType.SET_ZERO_POSITION,
                bus.host_id,
                motor_id,
                data,
            )
            bus.receive_status_frame(joint_name)
            if save_parameters:
                bus.transmit(
                    CommunicationType.SAVE_PARAMETERS,
                    bus.host_id,
                    motor_id,
                    bytes(8),
                )
                acknowledged, detail = receive_optional_status(bus, joint_name)
                if not acknowledged:
                    raise RuntimeError(f"save-parameters failed: {detail}")
            completed.append(joint_name)
            print(f"ZERO OK  {joint_name:20s} id=0x{motor_id:02X}")
        except Exception as exc:
            failures.append((joint_name, str(exc)))
            print(f"ZERO FAIL {joint_name:20s} id=0x{motor_id:02X}: {exc}")
    return completed, failures


def main():
    parser = argparse.ArgumentParser(
        description="Official RobStride SDK hardware-zero and reboot verifier"
    )
    parser.add_argument("--can-count", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["can0", "can1"],
        help="SocketCAN channels: 1=can0, 2=front back, 4=FR FL BR BL",
    )
    parser.add_argument("--model", default="rs-04")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument(
        "--motor-id",
        type=lambda value: int(value, 0),
        default=None,
        help="single motor ID in decimal or 0x-prefixed hexadecimal",
    )
    parser.add_argument(
        "--motor-name",
        default="test_motor",
        help="display name used with --motor-id",
    )
    parser.add_argument("--verify-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--save-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send official protocol type 22 after type 6 zero (default: true)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="read raw official parameters once and exit; use after power cycling",
    )
    parser.add_argument(
        "--watch-rate",
        type=float,
        default=0.0,
        help="repeat read-only snapshots at this rate in Hz; 0 reads once",
    )
    args = parser.parse_args()
    if not np.isfinite(args.watch_rate) or args.watch_rate < 0.0:
        parser.error("--watch-rate must be finite and >= 0")
    if args.watch_rate > 0.0:
        args.read_only = True

    motor_ids = load_motor_ids()
    if args.motor_id is not None:
        if args.motor_id <= 0 or args.motor_id > 0xFF:
            parser.error("--motor-id must be within 1..255")
        if args.can_count != 1 or len(args.channels) != 1:
            parser.error(
                "single-motor mode requires --can-count 1 and exactly one --channels value"
            )
        motor_ids = {str(args.motor_name): int(args.motor_id)}
    try:
        buses, joint_bus, channels_by_bus = build_buses(
            motor_ids=motor_ids,
            can_count=args.can_count,
            channels=args.channels,
            model=args.model,
            bitrate=args.bitrate,
        )
    except ValueError as exc:
        parser.error(str(exc))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / f"robstride_official_zero_{timestamp}.csv"
    connected_buses = []
    try:
        for bus_name, bus in buses.items():
            print(f"Opening {bus_name}: SocketCAN {channels_by_bus[bus_name]}")
            bus.connect(handshake=False)
            connected_buses.append(bus)

        before = read_snapshot(
            buses,
            joint_bus,
            motor_ids,
            channels_by_bus,
            phase="read_only" if args.read_only else "before_zero",
        )
        append_csv(log_path, before)
        connected = print_snapshot(before)
        print(f"CSV log: {log_path}")
        if args.read_only:
            if args.watch_rate > 0.0:
                period = 1.0 / args.watch_rate
                try:
                    while True:
                        time.sleep(period)
                        watched = read_snapshot(
                            buses,
                            joint_bus,
                            motor_ids,
                            channels_by_bus,
                            phase="watch",
                        )
                        append_csv(log_path, watched)
                        print_snapshot(watched)
                except KeyboardInterrupt:
                    print("\nStopped encoder watch.")
            return 0 if connected == len(before) else 2
        if connected != len(before):
            print("REFUSED: every motor must return MECHANICAL_POSITION before set-zero.")
            return 2

        choice = input(
            "\nPhysically place every joint at the intended zero pose. "
            "Enter 's' to hardware-zero ALL motors, or 'q' to quit: "
        ).strip().lower()
        if choice != "s":
            print("No zero command sent.")
            return 0

        completed, failures = set_zero_all(
            buses,
            joint_bus,
            motor_ids,
            save_parameters=bool(args.save_parameters),
        )
        time.sleep(0.25)
        after = read_snapshot(
            buses,
            joint_bus,
            motor_ids,
            channels_by_bus,
            phase="after_zero",
        )
        append_csv(log_path, after)
        print("\nOfficial raw parameters after zero:")
        print_snapshot(after)

        bad = [
            row for row in after
            if row["mechanical_position"] is None
            or abs(row["mechanical_position"]) > float(args.verify_tolerance)
        ]
        if failures or len(completed) != len(after) or bad:
            print("ZERO VERIFICATION FAILED. Do not enable motor control.")
            for row in bad:
                print(
                    f"  {row['joint_name']}: mechanical_position="
                    f"{row['mechanical_position']}"
                )
            return 3

        print("ZERO VERIFIED for all motors in the current power session.")
        print("Power-cycle motors, then rerun this script with --read-only.")
        return 0
    finally:
        for bus in reversed(connected_buses):
            try:
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                print(f"Disconnect warning: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
