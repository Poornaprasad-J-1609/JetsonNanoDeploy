#!/usr/bin/env python3
"""
Hardware-free CAN routing checker for GRALLATOR deployment.

This validates that:
  - every policy joint has a motor ID and explicit CAN bus assignment,
  - all command builders keep the correct one/two/four-CAN bus_name,
  - send paths actually route packets to the selected bus object,
  - MIT feedback frames are mapped by (bus, motor_id), so duplicate IDs on
    separate CAN networks can still be decoded correctly.
"""
import argparse
import time
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from motor_command_layer import MotorCommandLayer, float_to_uint
from robstride_can_interface import CanFrame
from state_estimator import MitFeedbackStateEstimator
from can_topology import bus_names_for_count, resolve_joint_can_bus


ROOT = Path(__file__).resolve().parents[1]


class FakeCanBus:
    def __init__(self, name):
        self.name = name
        self.sent_raw = []
        self.sent_signal = []
        self.frames = []
        self.read_count = 0

    def send_raw(self, can_id, data=b""):
        packet = (int(can_id), bytes(data))
        self.sent_raw.append(packet)
        return packet

    def send_signal_frame(self, motor_id):
        motor_id = int(motor_id)
        self.sent_signal.append(motor_id)
        return motor_id

    def read_available_frames(self, timeout=0.0, max_frames=256):
        self.read_count += 1
        frames = self.frames[:max_frames]
        del self.frames[:max_frames]
        return frames


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_policy_order():
    return list(load_yaml(ROOT / "config" / "joint_map.yaml")["policy_to_real_order"])


def load_motor_config():
    return load_yaml(ROOT / "config" / "motor_ids.yaml")


def can_comm_type(can_id):
    return (int(can_id) >> 24) & 0x1F


def motor_id_from_command_can_id(can_id):
    return int(can_id) & 0xFF


def make_feedback_frame(bus_name, motor_id, proto, q_motor, velocity=0.0, torque=0.0, temp_c=25.0):
    motor_id = int(motor_id)
    extra = motor_id & 0xFF
    can_id = (
        (int(proto.get("comm_type_feedback", 2)) << 24)
        | (extra << 8)
        | int(proto.get("master_id", 0xFD))
    )
    data = (
        float_to_uint(q_motor, proto["p_min"], proto["p_max"], 16).to_bytes(2, "big")
        + float_to_uint(velocity, proto["v_min"], proto["v_max"], 16).to_bytes(2, "big")
        + float_to_uint(torque, proto["tau_min"], proto["tau_max"], 16).to_bytes(2, "big")
        + int(round(float(temp_c) * 10.0)).to_bytes(2, "big")
    )
    frame = CanFrame(can_id=can_id, data=data, timestamp=time.monotonic())
    frame.bus_name = bus_name
    return frame


def validate_motor_config(policy_order, motor_ids, joint_can_bus, valid_bus_names):
    errors = []

    for joint_name in policy_order:
        if joint_name not in motor_ids:
            errors.append(f"missing motor_id for {joint_name}")
        if joint_name not in joint_can_bus:
            errors.append(f"missing joint_can_bus for {joint_name}")
        elif joint_can_bus[joint_name] not in valid_bus_names:
            errors.append(
                f"{joint_name} has invalid CAN bus '{joint_can_bus[joint_name]}'; "
                f"expected one of {sorted(valid_bus_names)}"
            )

    seen_bus_motor = {}
    for joint_name in policy_order:
        if joint_name not in motor_ids:
            continue
        bus_name = joint_can_bus.get(joint_name, "front")
        motor_id = int(motor_ids[joint_name])
        key = (bus_name, motor_id)
        if key in seen_bus_motor:
            errors.append(
                f"{joint_name} and {seen_bus_motor[key]} both use motor 0x{motor_id:02X} "
                f"on CAN bus '{bus_name}'"
            )
        seen_bus_motor[key] = joint_name

    if errors:
        raise RuntimeError("Invalid CAN/motor config:\n  - " + "\n  - ".join(errors))


def expected_counts_by_bus(commands):
    counts = {}
    for cmd in commands:
        counts[cmd.get("bus_name", "front")] = counts.get(cmd.get("bus_name", "front"), 0) + 1
    return counts


def format_counts(counts, bus_names):
    return " ".join(f"{bus_name}={int(counts.get(bus_name, 0))}" for bus_name in bus_names)


def make_fake_buses(bus_names):
    return {bus_name: FakeCanBus(bus_name) for bus_name in bus_names}


def assert_raw_routing(layer, commands, label, bus_names, expected_comm_type=None):
    buses = make_fake_buses(bus_names)
    expected = expected_counts_by_bus(commands)
    sent = layer.send_raw_commands(buses, commands)

    if len(sent) != len(commands):
        raise AssertionError(f"{label}: sent {len(sent)} packets for {len(commands)} commands")

    for bus_name, count in expected.items():
        actual = len(buses[bus_name].sent_raw)
        if actual != count:
            raise AssertionError(f"{label}: {bus_name} got {actual} raw packets, expected {count}")

    for cmd in commands:
        if cmd.get("bus_name", "front") not in buses:
            raise AssertionError(f"{label}: invalid bus_name in command {cmd}")
        if motor_id_from_command_can_id(cmd["can_id"]) != int(cmd["motor_id"]):
            raise AssertionError(f"{label}: CAN ID motor byte does not match command motor_id")
        if expected_comm_type is not None and can_comm_type(cmd["can_id"]) != expected_comm_type:
            raise AssertionError(
                f"{label}: comm_type {can_comm_type(cmd['can_id'])}, expected {expected_comm_type}"
            )

    return expected


def assert_mit_routing(layer, commands, bus_names):
    buses = make_fake_buses(bus_names)
    expected = expected_counts_by_bus(commands)
    sent = layer.send_signal_commands(buses, commands)

    if len(sent) != len(commands):
        raise AssertionError(f"MIT commands: sent {len(sent)} packets for {len(commands)} commands")

    for bus_name, count in expected.items():
        actual = len(buses[bus_name].sent_raw)
        if actual != count:
            raise AssertionError(f"MIT commands: {bus_name} got {actual} packets, expected {count}")

    return expected


def assert_signal_routing(layer, commands, bus_names):
    buses = make_fake_buses(bus_names)
    expected = expected_counts_by_bus(commands)
    layer.send_harmless_frames(buses, commands)

    for bus_name, count in expected.items():
        actual = len(buses[bus_name].sent_signal)
        if actual != count:
            raise AssertionError(f"signal frames: {bus_name} got {actual} packets, expected {count}")


def assert_shared_bus_read_once(layer, bus_names):
    shared = FakeCanBus("shared")
    first_bus = bus_names[0]
    shared.frames.append(make_feedback_frame(first_bus, 0x01, layer.proto, q_motor=0.0))
    buses = {bus_name: shared for bus_name in bus_names}
    frames = MotorCommandLayer.read_all_frames(buses, timeout=0.0)

    if shared.read_count != 1:
        raise AssertionError(f"shared physical CAN adapter was read {shared.read_count} times")
    if len(frames) != 1:
        raise AssertionError(f"shared physical CAN adapter returned {len(frames)} frames, expected 1")
    if getattr(frames[0], "bus_name", None) != first_bus:
        raise AssertionError("shared bus frame should keep the first logical bus tag")


def assert_duplicate_id_feedback_mapping(policy_order, motor_ids, joint_can_bus, layer):
    bus_to_joints = {}
    for name in policy_order:
        bus_to_joints.setdefault(joint_can_bus.get(name, "front"), []).append(name)

    populated_buses = [bus_name for bus_name, names in bus_to_joints.items() if names]
    if len(populated_buses) < 2:
        return False

    bus_a, bus_b = populated_buses[:2]
    joint_a = bus_to_joints[bus_a][0]
    joint_b = bus_to_joints[bus_b][0]

    duplicate_motor_ids = dict(motor_ids)
    duplicate_motor_ids[joint_b] = int(duplicate_motor_ids[joint_a])
    duplicate_layer = MotorCommandLayer(
        policy_order=policy_order,
        motor_ids=duplicate_motor_ids,
        active_joints=policy_order,
        joint_can_bus=joint_can_bus,
    )
    estimator = MitFeedbackStateEstimator(
        q_initial=np.zeros(len(policy_order), dtype=np.float32),
        policy_order=policy_order,
        motor_ids=duplicate_motor_ids,
        motor_layer=duplicate_layer,
        bus=None,
        imu_sensor=None,
    )

    q_a = 0.123
    q_b = -0.234
    offset_a = float(duplicate_layer.joint_offsets[joint_a])
    offset_b = float(duplicate_layer.joint_offsets[joint_b])
    motor_id = int(duplicate_motor_ids[joint_a])
    frames = [
        make_feedback_frame(bus_a, motor_id, duplicate_layer.proto, q_a + offset_a),
        make_feedback_frame(bus_b, motor_id, duplicate_layer.proto, q_b + offset_b),
    ]
    count = estimator.update_from_frames(frames)
    if count != 2:
        raise AssertionError(f"duplicate ID feedback test decoded {count} frames, expected 2")

    index_a = policy_order.index(joint_a)
    index_b = policy_order.index(joint_b)
    if abs(float(estimator.q_current[index_a]) - q_a) > 0.003:
        raise AssertionError("first-bus duplicate-ID feedback mapped to wrong joint/value")
    if abs(float(estimator.q_current[index_b]) - q_b) > 0.003:
        raise AssertionError("second-bus duplicate-ID feedback mapped to wrong joint/value")
    return True


def assert_software_zero_calibration(policy_order, motor_ids, joint_can_bus):
    joint = "FR_thigh_joint" if "FR_thigh_joint" in policy_order else policy_order[0]
    index = policy_order.index(joint)
    bus_name = joint_can_bus.get(joint, "front")
    motor_id = int(motor_ids[joint])

    layer = MotorCommandLayer(
        policy_order=policy_order,
        motor_ids=motor_ids,
        active_joints=[joint],
        joint_can_bus=joint_can_bus,
    )
    estimator = MitFeedbackStateEstimator(
        q_initial=np.zeros(len(policy_order), dtype=np.float32),
        policy_order=policy_order,
        motor_ids=motor_ids,
        motor_layer=layer,
        bus=None,
        imu_sensor=None,
    )

    raw_crouch = 0.73
    estimator.update_from_frames([
        make_feedback_frame(bus_name, motor_id, layer.proto, raw_crouch)
    ])
    if abs(float(estimator.q_current[index]) - raw_crouch) > 0.004:
        raise AssertionError("raw feedback should be used before software zero calibration")

    updated, missing = estimator.apply_software_zero(active_joints=[joint])
    if missing or joint not in updated:
        raise AssertionError("software zero calibration did not update the active joint")
    if abs(float(estimator.q_current[index])) > 0.004:
        raise AssertionError("software zero calibration did not make current q equal zero")

    q_target = np.zeros(len(policy_order), dtype=np.float32)
    q_target[index] = 0.120
    commands = layer.build_mit_commands(
        q_target,
        phase="policy",
        feedback_by_joint=estimator.last_feedback_by_joint,
    )
    expected_p = raw_crouch + float(q_target[index])
    if len(commands) != 1 or abs(float(commands[0]["p_des"]) - expected_p) > 0.010:
        raise AssertionError("MIT command did not remain continuous after software zero")

    reverse_layer = MotorCommandLayer(
        policy_order=policy_order,
        motor_ids=motor_ids,
        active_joints=[joint],
        joint_can_bus=joint_can_bus,
    )
    reverse_layer.joint_directions[joint] = -1.0
    reverse_estimator = MitFeedbackStateEstimator(
        q_initial=np.zeros(len(policy_order), dtype=np.float32),
        policy_order=policy_order,
        motor_ids=motor_ids,
        motor_layer=reverse_layer,
        bus=None,
        imu_sensor=None,
    )
    reverse_estimator.update_from_frames([
        make_feedback_frame(bus_name, motor_id, reverse_layer.proto, raw_crouch)
    ])
    reverse_estimator.apply_software_zero(active_joints=[joint])
    reverse_estimator.update_from_frames([
        make_feedback_frame(bus_name, motor_id, reverse_layer.proto, raw_crouch - 0.1)
    ])
    if abs(float(reverse_estimator.q_current[index]) - 0.1) > 0.006:
        raise AssertionError("joint_direction=-1 did not invert encoder feedback consistently")


def q_midpoint_from_limits(layer, policy_order):
    return np.array(
        [
            0.5 * (float(layer.hard_joint_limits[name][0]) + float(layer.hard_joint_limits[name][1]))
            for name in policy_order
        ],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser(description="Dry CAN routing check; no serial ports are opened.")
    parser.add_argument(
        "--can-count",
        type=int,
        choices=[1, 2, 4],
        default=2,
        help="1=all joints on one bus, 2=front/back, 4=FR/FL/BR/BL",
    )
    parser.add_argument(
        "--active-joints",
        nargs="*",
        default=None,
        help="optional subset to route-test; config validity is still checked for all policy joints",
    )
    args = parser.parse_args()

    policy_order = load_policy_order()
    motor_cfg = load_motor_config()
    motor_ids = motor_cfg["motor_ids"]
    joint_can_bus = resolve_joint_can_bus(policy_order, args.can_count)
    bus_names = bus_names_for_count(args.can_count)
    validate_motor_config(policy_order, motor_ids, joint_can_bus, set(bus_names))

    active_joints = args.active_joints if args.active_joints is not None else policy_order
    layer = MotorCommandLayer(
        policy_order=policy_order,
        motor_ids=motor_ids,
        active_joints=active_joints,
        joint_can_bus=joint_can_bus,
    )

    q_target = q_midpoint_from_limits(layer, policy_order)
    mit_commands = layer.build_mit_commands(q_target, phase="policy")
    mit_counts = assert_mit_routing(layer, mit_commands, bus_names)
    assert_signal_routing(layer, mit_commands, bus_names)

    proto = layer.proto
    enable_counts = assert_raw_routing(
        layer,
        layer.build_enable_commands(),
        "enable commands",
        bus_names,
        expected_comm_type=int(proto["comm_type_enable"]),
    )
    poll_counts = assert_raw_routing(
        layer,
        layer.build_feedback_poll_commands(),
        "feedback poll commands",
        bus_names,
        expected_comm_type=int(proto["comm_type_stop"]),
    )
    zero_counts = assert_raw_routing(
        layer,
        layer.build_set_zero_commands(),
        "set-zero commands",
        bus_names,
        expected_comm_type=int(proto["comm_type_set_zero"]),
    )
    stop_counts = assert_raw_routing(
        layer,
        layer.build_stop_commands(),
        "stop commands",
        bus_names,
        expected_comm_type=int(proto["comm_type_stop"]),
    )
    assert_shared_bus_read_once(layer, bus_names)
    duplicate_mapping_checked = assert_duplicate_id_feedback_mapping(
        policy_order,
        motor_ids,
        joint_can_bus,
        layer,
    )
    assert_software_zero_calibration(policy_order, motor_ids, joint_can_bus)

    bus_to_joints = {
        bus_name: [name for name in policy_order if joint_can_bus.get(name, "front") == bus_name]
        for bus_name in bus_names
    }

    print("CAN routing check OK.")
    print("Configured buses:")
    for bus_name, joints in bus_to_joints.items():
        joined = ", ".join(f"{name}=0x{int(motor_ids[name]):02X}" for name in joints)
        print(f"  {bus_name:5s}: {len(joints)} joint(s) -> {joined}")
    print("Route-tested active joints:", ", ".join(layer.active_joints))
    print("MIT command routing:      ", format_counts(mit_counts, bus_names))
    print("Enable command routing:   ", format_counts(enable_counts, bus_names))
    print("Feedback poll routing:    ", format_counts(poll_counts, bus_names))
    print("Set-zero command routing: ", format_counts(zero_counts, bus_names))
    print("Stop command routing:     ", format_counts(stop_counts, bus_names))
    print("Shared-port de-duplication: OK")
    if duplicate_mapping_checked:
        print("Duplicate motor IDs on separate CAN buses: feedback mapping OK")
    else:
        print("Duplicate motor IDs on separate CAN buses: skipped for one-CAN topology")
    print("Software zero calibration and joint direction handling: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
