#!/usr/bin/env python3
"""Passively benchmark a SocketCAN command/feedback cycle with stop frames."""

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
    bus_names_for_count,
    close_can_buses,
    open_can_buses,
    resolve_joint_can_bus,
)
from motor_command_layer import MotorCommandLayer  # noqa: E402
from policy_runner import EXPECTED_POLICY_JOINT_ORDER  # noqa: E402


def percentile(values, value):
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def summarize_transport(tx_times, feedback_times, missed_batches, backpressure_events):
    if not tx_times:
        raise ValueError("no CAN batches were measured")
    return {
        "median_batch_tx_ms": 1000.0 * statistics.median(tx_times),
        "p95_batch_tx_ms": 1000.0 * percentile(tx_times, 95),
        "p99_batch_tx_ms": 1000.0 * percentile(tx_times, 99),
        "maximum_batch_tx_ms": 1000.0 * max(tx_times),
        "feedback_completion_latency_ms": (
            0.0 if not feedback_times else 1000.0 * percentile(feedback_times, 95)
        ),
        "missed_batches": int(missed_batches),
        "socket_backpressure_events": int(backpressure_events),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--can-count", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--can-interface", default="slcan0")
    parser.add_argument(
        "--can-interfaces",
        nargs="+",
        default=None,
        help="ordered interfaces: one=can0; two=front back; four=FR FL BR BL",
    )
    parser.add_argument("--motors", type=int, default=12)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.motors <= 12:
        parser.error("--motors must be within 1..12")
    if args.rate <= 0.0 or args.duration <= 0.0:
        parser.error("--rate and --duration must be > 0")
    bus_names = bus_names_for_count(args.can_count)
    interfaces = args.can_interfaces
    if interfaces is None:
        interfaces = (
            [args.can_interface]
            if args.can_count == 1
            else [f"slcan{index}" for index in range(args.can_count)]
        )
    if len(interfaces) != len(bus_names):
        parser.error(
            f"--can-count {args.can_count} requires {len(bus_names)} "
            f"--can-interfaces values ({', '.join(bus_names)})"
        )
    if args.can_count > 1 and args.motors != 12:
        parser.error("multi-adapter qualification requires --motors 12")

    motor_cfg = yaml.safe_load((ROOT / "config" / "motor_ids.yaml").read_text())
    active = list(EXPECTED_POLICY_JOINT_ORDER[: args.motors])
    joint_can_bus = resolve_joint_can_bus(EXPECTED_POLICY_JOINT_ORDER, args.can_count)
    layer = MotorCommandLayer(
        list(EXPECTED_POLICY_JOINT_ORDER),
        motor_cfg["motor_ids"],
        active_joints=active,
        joint_can_bus=joint_can_bus,
    )
    buses = open_can_buses(
        dict(zip(bus_names, interfaces)),
        backend="socketcan",
        bitrate=args.bitrate,
        timeout=min(0.01, 0.5 / args.rate),
    )
    commands = layer.build_feedback_poll_commands()
    expected_bus_motor_ids = {
        (joint_can_bus[name], int(motor_cfg["motor_ids"][name]))
        for name in active
    }
    tx_times = []
    feedback_times = []
    missed = 0
    dt = 1.0 / args.rate
    deadline = time.monotonic()
    end = deadline + args.duration
    unique_buses = {id(bus): bus for bus in buses.values()}.values()
    backpressure_start = sum(
        int(getattr(bus, "backpressure_events", 0)) for bus in unique_buses
    )
    try:
        while time.monotonic() < end:
            cycle = time.monotonic()
            layer.send_raw_commands(buses, commands)
            tx_times.append(time.monotonic() - cycle)
            feedback_start = time.monotonic()
            frames = MotorCommandLayer.read_all_frames(
                buses,
                timeout=max(0.0, dt - (feedback_start - cycle)),
                expected_bus_motor_ids=expected_bus_motor_ids,
            )
            feedback_times.append(time.monotonic() - feedback_start)
            received = {
                (
                    str(frame.bus_name),
                    ((int(frame.can_id) >> 8) & 0xFFFF) & 0xFF,
                )
                for frame in frames
            }
            if not expected_bus_motor_ids.issubset(received):
                missed += 1
            if args.details:
                for bus_name, bus in buses.items():
                    for item in getattr(bus, "last_frame_timings", []):
                        print(
                            f"bus={bus_name:<5s} frame={item['frame_index']:02d} "
                            f"motor=0x{item['motor_id']:02X} "
                            f"send={1000.0 * item['send_duration_s']:.3f}ms "
                            f"txqlen={item['socket_tx_queue_len']} "
                            f"EAGAIN={item['eagain_count']}"
                        )
            deadline += dt
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        layer.close()
        close_can_buses(buses)

    unique_buses = {id(bus): bus for bus in buses.values()}.values()
    backpressure_events = sum(
        int(getattr(bus, "backpressure_events", 0)) for bus in unique_buses
    ) - backpressure_start
    result = summarize_transport(
        tx_times,
        feedback_times,
        missed,
        backpressure_events,
    )
    print(
        "CAN CONTROL-CYCLE BENCHMARK "
        f"({args.can_count} adapter(s), passive stop/poll frames)"
    )
    print(f"deadline_ms: {1000.0 * dt:.3f}")
    for key, value in result.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    if (
        result["p99_batch_tx_ms"] > 1000.0 * dt
        or result["missed_batches"] > 0
        or result["socket_backpressure_events"] > 0
    ):
        print(f"{args.can_count}-ADAPTER TRANSPORT QUALIFICATION FAILED")
        return 1
    print(f"{args.can_count}-ADAPTER TRANSPORT QUALIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
