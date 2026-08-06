import time

import numpy as np

from can_topology import resolve_joint_can_bus
from main_controller import (
    feedback_recency_summary,
    fresh_feedback_by_joint,
    refresh_active_feedback_before_fault,
    refresh_estimator_feedback,
)
from motor_command_layer import MotorCommandLayer
from policy_runner import PolicyRunner
from robstride_can_interface import SocketCan


def feedback_can_id(motor_id, comm_type=2):
    return (int(comm_type) << 24) | ((int(motor_id) & 0xFF) << 8) | 0xFD


def test_active_can_streamer_owns_feedback_socket_reads():
    class Streamer:
        has_active_commands = True

        @staticmethod
        def drain_received():
            return ["cached-frame"]

    class Estimator:
        can_feedback_streamer = Streamer()

        def __init__(self):
            self.direct_reads = 0
            self.frames = []

        def update_from_frames(self, frames):
            self.frames.extend(frames)
            return len(frames)

        def refresh_from_bus(self, timeout=0.0, expected_bus_motor_ids=None):
            self.direct_reads += 1
            return 99

    estimator = Estimator()
    assert refresh_estimator_feedback(estimator, timeout=0.003) == 1
    assert estimator.frames == ["cached-frame"]
    assert estimator.direct_reads == 0


class FakeCanMessage:
    def __init__(self, motor_id, comm_type=2):
        self.arbitration_id = feedback_can_id(motor_id, comm_type=comm_type)
        self.data = bytes(8)
        self.is_extended_id = True


class FakeRecvBus:
    def __init__(self, messages, fail_on_extra=False):
        self.messages = list(messages)
        self.fail_on_extra = bool(fail_on_extra)
        self.recv_timeouts = []

    def recv(self, timeout=0.0):
        self.recv_timeouts.append(timeout)
        if self.messages:
            return self.messages.pop(0)
        if self.fail_on_extra:
            raise AssertionError("feedback reader waited after all expected IDs arrived")
        return None


def socketcan_with_fake_bus(messages, fail_on_extra=False):
    transport = SocketCan(channel="slcan0")
    transport.bus = FakeRecvBus(messages, fail_on_extra=fail_on_extra)
    return transport


def test_socketcan_transport_state_exists_before_filter_configuration():
    transport = SocketCan(channel="slcan0")
    transport.send_raw = lambda can_id, data=b"": (int(can_id), bytes(data))

    assert transport.send_raw_sequence([(0x101, b"\x01")]) == [
        (0x101, b"\x01")
    ]
    transport.close()


def test_feedback_filter_configuration_does_not_reset_transport_state():
    class FilterBus:
        def __init__(self):
            self.filters = None

        def set_filters(self, filters):
            self.filters = filters

    transport = SocketCan(channel="slcan0")
    original_lock = transport._send_lock
    transport.backpressure_events = 7
    transport.bus = FilterBus()

    transport.configure_feedback_filters()

    assert transport.bus.filters
    assert transport._send_lock is original_lock
    assert transport.backpressure_events == 7


def test_feedback_collector_returns_immediately_after_all_expected_ids():
    transport = socketcan_with_fake_bus(
        [FakeCanMessage(1), FakeCanMessage(2)],
        fail_on_extra=True,
    )

    frames = transport.read_available_frames(
        timeout=0.5,
        expected_motor_ids={1, 2},
        feedback_comm_types={2, 24},
    )

    assert len(frames) == 2
    assert len(transport.bus.recv_timeouts) == 2


def test_feedback_collector_does_not_wait_full_timeout_after_complete_feedback():
    transport = socketcan_with_fake_bus(
        [FakeCanMessage(1), FakeCanMessage(2), FakeCanMessage(3)],
        fail_on_extra=True,
    )

    frames = transport.read_available_frames(
        timeout=0.25,
        expected_motor_ids={1, 2},
        feedback_comm_types={2, 24},
    )

    assert [((frame.can_id >> 8) & 0xFF) for frame in frames] == [1, 2]
    assert len(transport.bus.recv_timeouts) == 2


def test_motor_layer_feedback_collector_annotates_bus_and_early_exits():
    transport = socketcan_with_fake_bus(
        [FakeCanMessage(10), FakeCanMessage(11)],
        fail_on_extra=True,
    )

    frames = MotorCommandLayer.read_all_frames(
        {"can0": transport},
        timeout=0.25,
        expected_bus_motor_ids={("can0", 10), ("can0", 11)},
        proto={"comm_type_feedback": 2, "comm_type_active_feedback": 24},
    )

    assert len(frames) == 2
    assert {frame.bus_name for frame in frames} == {"can0"}
    assert len(transport.bus.recv_timeouts) == 2


def test_previous_cycle_feedback_remains_usable_inside_freshness_window():
    now = time.monotonic()
    estimator = type(
        "Estimator",
        (),
        {
            "last_command_send_timestamp": now - 0.005,
            "last_feedback_by_joint": {
                "FR_calf_joint": {
                    "timestamp": now - 0.015,
                    "joint_position": 0.0,
                    "joint_velocity": 0.0,
                }
            },
        },
    )()

    fresh, missing = fresh_feedback_by_joint(
        estimator,
        ["FR_calf_joint"],
        max_age_s=0.04,
    )
    recency = feedback_recency_summary(
        estimator,
        ["FR_calf_joint"],
        max_age_s=0.04,
    )

    assert "FR_calf_joint" in fresh
    assert missing == []
    assert recency["fresh_previous_cycle"] == 1
    assert recency["fresh_current_cycle"] == 0


def test_stale_feedback_is_reported_for_policy_freeze_path():
    estimator = type(
        "Estimator",
        (),
        {
            "last_command_send_timestamp": time.monotonic(),
            "last_feedback_by_joint": {
                "FR_calf_joint": {
                    "timestamp": time.monotonic() - 0.5,
                    "joint_position": 0.0,
                    "joint_velocity": 0.0,
                }
            },
        },
    )()

    fresh, missing = fresh_feedback_by_joint(
        estimator,
        ["FR_calf_joint"],
        max_age_s=0.04,
    )
    recency = feedback_recency_summary(
        estimator,
        ["FR_calf_joint"],
        max_age_s=0.04,
    )

    assert fresh == {}
    assert missing == ["FR_calf_joint"]
    assert recency["stale"] == 1


class MissingFeedbackEstimator:
    def __init__(self):
        self.q_current = np.zeros(2, dtype=np.float32)
        self.last_feedback_by_joint = {}
        self.refresh_calls = []

    def expected_feedback_bus_motor_ids(self, active_joints=None):
        return {("can0", index + 1) for index, _ in enumerate(active_joints or [])}

    def refresh_from_bus(self, timeout=0.0, expected_bus_motor_ids=None):
        self.refresh_calls.append((timeout, set(expected_bus_motor_ids or [])))
        return 0


class MinimalMotorLayer:
    active_joints = ["j0", "j1"]
    motor_ids = {"j0": 1, "j1": 2}
    joint_can_bus = {"j0": "can0", "j1": "can0"}


class MinimalSafety:
    max_feedback_age_s = 0.01


def test_missing_feedback_uses_bounded_recovery_timeout():
    estimator = MissingFeedbackEstimator()

    fresh, n_active = refresh_active_feedback_before_fault(
        estimator=estimator,
        motor_layer=MinimalMotorLayer(),
        safety=MinimalSafety(),
        buses=None,
        mode="mit-signal",
        feedback_timeout=0.001,
        max_wait_s=0.003,
        max_age_s=0.01,
    )

    assert fresh == 0
    assert n_active == 2
    assert estimator.refresh_calls
    assert max(timeout for timeout, _ in estimator.refresh_calls) <= 0.001


def test_all_motor_stop_frames_are_sent_to_one_socketcan_bus():
    runner = PolicyRunner()
    motor_ids = {
        name: index + 1
        for index, name in enumerate(runner.policy_order)
    }
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=runner.policy_order,
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 1),
    )

    class StopBus:
        requires_frame_gap = False

        def __init__(self):
            self.sent = []

        def send_raw_sequence(self, frames, frame_gap_s=0.0):
            self.sent.extend(frames)
            return frames

    bus = StopBus()
    commands = layer.build_stop_commands()
    layer.send_raw_commands({"can0": bus}, commands)

    assert len(commands) == 12
    assert len(bus.sent) == 12
    assert {command["motor_id"] for command in commands} == set(range(1, 13))


def test_two_can_command_lanes_send_complete_deterministic_batches():
    runner = PolicyRunner()
    motor_ids = {
        name: index + 1
        for index, name in enumerate(runner.policy_order)
    }
    layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=runner.policy_order,
        joint_can_bus=resolve_joint_can_bus(runner.policy_order, 2),
    )
    dispatch_order = []

    class RecordingBus:
        requires_frame_gap = False

        def __init__(self, name):
            self.name = name
            self.sent = []

        def send_raw_sequence(self, frames, frame_gap_s=0.0):
            dispatch_order.append(self.name)
            self.sent.extend(frames)
            return frames

    front = RecordingBus("front")
    back = RecordingBus("back")
    commands = layer.build_stop_commands()
    try:
        sent = layer.send_raw_commands({"front": front, "back": back}, commands)
    finally:
        layer.close()

    assert len(dispatch_order) == 2
    assert set(dispatch_order) == {"front", "back"}
    assert len(sent) == 12
    assert sent == [(command["can_id"], command["data"]) for command in commands]
    assert len(front.sent) == 6
    assert len(back.sent) == 6
