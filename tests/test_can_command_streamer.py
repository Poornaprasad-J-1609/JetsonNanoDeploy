import threading
import time

from can_command_streamer import CanCommandStreamer


def wait_until(predicate, timeout=0.5):
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


def test_default_stale_timeout_tolerates_brief_jetson_scheduler_stalls():
    streamer = CanCommandStreamer(send_callback=lambda _commands: None)
    assert streamer.command_dt_s == 0.005
    assert streamer.stale_timeout_s == 0.250


def test_streamer_repeats_only_latest_atomic_snapshot_at_200_hz():
    sent = []
    lock = threading.Lock()

    def record(commands):
        with lock:
            sent.append(tuple(command["target"] for command in commands))

    streamer = CanCommandStreamer(
        send_callback=record,
        command_dt_s=0.005,
        stale_timeout_s=0.100,
        fault_consecutive_overruns=3,
    )
    streamer.start()
    try:
        streamer.submit([{"target": "old-a"}, {"target": "old-b"}])
        assert wait_until(lambda: len(sent) >= 2)

        streamer.submit([{"target": "new-a"}, {"target": "new-b"}])
        assert wait_until(lambda: ("new-a", "new-b") in sent)
        time.sleep(0.012)

        with lock:
            first_new = sent.index(("new-a", "new-b"))
            after_replacement = list(sent[first_new:])
        assert after_replacement
        assert all(item == ("new-a", "new-b") for item in after_replacement)
        assert streamer.fault_reason is None
        assert streamer.telemetry()["can_command_dt_ms"] == 5.0
    finally:
        streamer.stop()


def test_streamer_faults_and_clears_a_stale_policy_target():
    sent = []
    streamer = CanCommandStreamer(
        send_callback=lambda commands: sent.append(tuple(commands)),
        command_dt_s=0.005,
        stale_timeout_s=0.020,
        fault_consecutive_overruns=3,
    )
    streamer.start()
    try:
        streamer.submit([{"target": 1.0}])
        assert wait_until(lambda: len(sent) >= 2)
        assert wait_until(lambda: streamer.fault_reason is not None)
        count_at_fault = len(sent)
        time.sleep(0.015)

        assert "became stale" in streamer.fault_reason
        assert len(sent) == count_at_fault
        assert streamer.telemetry()["can_command_stale_events"] == 1
    finally:
        streamer.stop()


def test_streamer_faults_after_consecutive_five_ms_batch_overruns():
    send_count = 0

    def slow_send(_commands):
        nonlocal send_count
        send_count += 1
        time.sleep(0.007)

    streamer = CanCommandStreamer(
        send_callback=slow_send,
        command_dt_s=0.005,
        stale_timeout_s=0.100,
        fault_consecutive_overruns=3,
        transport_label="2-ADAPTER",
    )
    streamer.start()
    try:
        streamer.submit([{"target": 1.0}])
        assert wait_until(lambda: streamer.fault_reason is not None)

        assert send_count == 3
        assert "2-ADAPTER TRANSPORT DEADLINE FAILED" in streamer.fault_reason
        telemetry = streamer.telemetry()
        assert telemetry["can_command_consecutive_overruns"] == 3
        assert telemetry["can_command_max_batch_ms"] > 5.0
        assert "can_command_scheduler_lateness_ms" in telemetry
    finally:
        streamer.stop()


def test_streamer_collects_feedback_for_the_policy_thread():
    pending = [["front-frame", "back-frame"]]

    def receive():
        return pending.pop(0) if pending else []

    streamer = CanCommandStreamer(
        send_callback=lambda _commands: None,
        receive_callback=receive,
        command_dt_s=0.005,
        stale_timeout_s=0.100,
    )
    streamer.start()
    try:
        streamer.submit([{"target": 1.0}])
        assert wait_until(
            lambda: streamer.telemetry()["can_feedback_receive_count"] >= 2
        )
        assert streamer.has_active_commands
        assert streamer.drain_received() == ["front-frame", "back-frame"]
        assert streamer.drain_received() == []
        streamer.clear()
        assert not streamer.has_active_commands
    finally:
        streamer.stop()


def test_streamer_can_collect_feedback_on_every_worker_cycle():
    receive_count = 0

    def receive():
        nonlocal receive_count
        receive_count += 1
        return []

    streamer = CanCommandStreamer(
        send_callback=lambda _commands: None,
        receive_callback=receive,
        send_only_on_change=True,
        receive_every_n_cycles=1,
        command_dt_s=0.005,
        stale_timeout_s=0.100,
    )
    streamer.start()
    try:
        streamer.submit([{"target": 1.0}])
        assert wait_until(lambda: receive_count >= 3)
        assert streamer.fault_reason is None
    finally:
        streamer.stop()


def test_streamer_calls_low_level_trace_callback_each_can_cycle():
    traced = []

    def trace_callback(**snapshot):
        traced.append(snapshot)

    streamer = CanCommandStreamer(
        send_callback=lambda _commands: None,
        cycle_callback=trace_callback,
        command_dt_s=0.005,
        stale_timeout_s=0.100,
    )
    streamer.start()
    try:
        streamer.submit([{"target": 1.0}])
        assert wait_until(lambda: len(traced) >= 3)
        assert [item["cycle_index"] for item in traced[:3]] == [1, 2, 3]
        assert all(item["commands"][0]["target"] == 1.0 for item in traced[:3])
        assert streamer.fault_reason is None
    finally:
        streamer.stop()
