#!/usr/bin/env python3
"""Configure the Xsens MTData2 stream required by the 48D locomotion policy."""

import argparse
import sys
import time
from pathlib import Path

import serial


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imu_interface import extract_xbus_packets  # noqa: E402


MID_GOTO_MEASUREMENT = 0x10
MID_GOTO_MEASUREMENT_ACK = 0x11
MID_GOTO_CONFIG = 0x30
MID_GOTO_CONFIG_ACK = 0x31
MID_SET_OUTPUT_CONFIGURATION = 0xC0
MID_SET_OUTPUT_CONFIGURATION_ACK = 0xC1

XDI_QUATERNION = 0x2010
XDI_RATE_OF_TURN = 0x8020


def xbus_frame(mid, payload=b"", bus_id=0xFF):
    payload = bytes(payload)
    if len(payload) >= 0xFF:
        raise ValueError("extended-length Xbus commands are not needed here")
    body = bytes([int(bus_id), int(mid), len(payload)]) + payload
    checksum = (-sum(body)) & 0xFF
    return bytes([0xFA]) + body + bytes([checksum])


def receive_mid(port, expected_mid, timeout=1.0):
    buffer = bytearray()
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        waiting = int(port.in_waiting)
        chunk = port.read(waiting if waiting > 0 else 1)
        if chunk:
            buffer.extend(chunk)
        for _, mid, payload in extract_xbus_packets(buffer):
            if mid == int(expected_mid):
                return bytes(payload)
        time.sleep(0.001)
    raise TimeoutError(f"Xsens acknowledgement MID 0x{expected_mid:02X} timed out")


def transact(port, mid, expected_mid, payload=b"", timeout=1.0):
    port.reset_input_buffer()
    port.write(xbus_frame(mid, payload))
    port.flush()
    return receive_mid(port, expected_mid, timeout=timeout)


def parse_output_configuration(payload):
    payload = bytes(payload)
    if len(payload) % 4 != 0:
        raise ValueError(f"invalid output-configuration payload length {len(payload)}")
    return [
        (
            int.from_bytes(payload[index:index + 2], "big"),
            int.from_bytes(payload[index + 2:index + 4], "big"),
        )
        for index in range(0, len(payload), 4)
    ]


def configuration_payload(entries):
    return b"".join(
        int(data_id).to_bytes(2, "big") + int(rate).to_bytes(2, "big")
        for data_id, rate in entries
    )


def format_configuration(entries):
    return ", ".join(f"0x{data_id:04X}@{rate}Hz" for data_id, rate in entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--rate", type=int, default=200)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write quaternion and rate-of-turn output configuration",
    )
    args = parser.parse_args()
    if args.rate <= 0 or args.rate > 2000:
        parser.error("--rate must be within 1..2000 Hz")

    desired = [
        (XDI_QUATERNION, int(args.rate)),
        (XDI_RATE_OF_TURN, int(args.rate)),
    ]
    port = serial.Serial(args.port, args.baud, timeout=0.02)
    in_config = False
    try:
        transact(port, MID_GOTO_CONFIG, MID_GOTO_CONFIG_ACK)
        in_config = True
        current_payload = transact(
            port,
            MID_SET_OUTPUT_CONFIGURATION,
            MID_SET_OUTPUT_CONFIGURATION_ACK,
        )
        current = parse_output_configuration(current_payload)
        print("Current Xsens output:", format_configuration(current))
        print("Policy Xsens output:", format_configuration(desired))
        if not args.apply:
            print("Inspection only; use --apply to write the policy output configuration.")
            return 0

        acknowledgement = transact(
            port,
            MID_SET_OUTPUT_CONFIGURATION,
            MID_SET_OUTPUT_CONFIGURATION_ACK,
            payload=configuration_payload(desired),
        )
        if acknowledgement:
            acknowledged = parse_output_configuration(acknowledgement)
            if acknowledged != desired:
                raise RuntimeError(
                    "Xsens acknowledged an unexpected output configuration: "
                    + format_configuration(acknowledged)
                )
        verify_payload = transact(
            port,
            MID_SET_OUTPUT_CONFIGURATION,
            MID_SET_OUTPUT_CONFIGURATION_ACK,
        )
        verified = parse_output_configuration(verify_payload)
        if verified != desired:
            raise RuntimeError(
                "Xsens output verification failed: " + format_configuration(verified)
            )
        print("Verified Xsens output:", format_configuration(verified))
        return 0
    finally:
        if in_config:
            try:
                transact(
                    port,
                    MID_GOTO_MEASUREMENT,
                    MID_GOTO_MEASUREMENT_ACK,
                )
            except Exception as exc:
                print(f"WARNING: could not restore measurement mode: {exc}")
        port.close()


if __name__ == "__main__":
    raise SystemExit(main())
